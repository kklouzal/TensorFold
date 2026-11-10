"""Current grouped-down/publication reuse after a separately supplied GU."""

from types import SimpleNamespace

import pytest
import torch

from tensorfold.cuda.sm70 import gguf_ops

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="SM70 owned down bridge")


@pytest.fixture(scope="module")
def prepared_banks():
    if torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("SM70 owned down bridge")
    device = torch.device("cuda", torch.cuda.current_device())
    prepared = gguf_ops.prepare_owned_from_activation(device)

    def gate_bank(scale):
        blocks = torch.full((256 * 160, 10, 136), 0x88, dtype=torch.uint8, device=device)
        blocks[:, :, :2] = torch.tensor([scale], dtype=torch.float16, device=device).view(torch.uint8)
        blocks[:, :, 2:4] = 0xAA
        blocks[:, :, 4:8] = 0x11
        return SimpleNamespace(data=blocks.reshape(256 * 160, 1360), ggml_type=23, column_offset=0)

    gate, up = gate_bank(256.), gate_bank(2. ** -24)
    down_banks = {}
    for kind, block, code in ((20, 32, 0x88), (42, 64, 0xAA)):
        count = 640 // block
        blocks = torch.full((256 * 2560, count, 18), code, dtype=torch.uint8, device=device)
        blocks[:, :, :2] = torch.arange(1, count + 1, dtype=torch.float16, device=device).view(torch.uint8).reshape(count, 2)
        down_banks[kind] = blocks.reshape(256 * 2560, count * 18)
    try:
        yield prepared, gate, up, down_banks
    finally:
        torch.cuda.synchronize(device)


def baseline(prepared_banks, kind, offset, rows, first, dtype=torch.float32):
    prepared, gate, up, down_banks = prepared_banks
    down = SimpleNamespace(data=down_banks[kind], ggml_type=kind, column_offset=offset)
    x = torch.zeros((rows, 2560), dtype=torch.float16, device="cuda")
    x[:, 0] = 256
    owned = [first + value for value in (0, 63, 127, 191, 255)]
    foreign = [256 - first + value for value in (0, 63, 127, 191, 255)]
    picks = torch.tensor([*owned, *foreign, 512], dtype=torch.int32, device="cuda").repeat(rows, 1)
    shared = torch.full((rows, 2560), -3.25, dtype=torch.float32, device="cuda")
    local = torch.empty((rows, 11), dtype=torch.int64, device="cuda")
    activation = torch.empty((rows * 11, 160), dtype=torch.float16, device="cuda")
    out = torch.empty((rows, 11, 2560), dtype=dtype, device="cuda")
    grouping = torch.empty(gguf_ops.grouped_workspace_words(rows * 11, 256), dtype=torch.int32, device="cuda")
    gguf_ops.experts_owned(x, picks, gate, up, down, first=first, experts=512, width=160,
                          shared=shared, local=local, activation=activation, out=out,
                          grouping=grouping, _gate_up=gguf_ops._GroupedGateUp(4))
    expected = out.clone()
    # Old GU has no status ABI. Shared activation is unused by down, so supply
    # the new explicit+0 status after the independent baseline completed GU.
    activation.reshape(rows, 11, 160)[:, 10].zero_()
    return prepared, (x, picks, gate, up, down), dict(first=first, shared=shared, local=local,
        activation=activation, out=out, grouping=grouping), expected


@pytest.mark.parametrize("kind,offset,rows,first", [(20, 0, 1, 0), (20, 160, 24, 256),
    (20, 480, 32, 0), (42, 0, 1, 256), (42, 160, 24, 0), (42, 480, 32, 256)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_exact_current_down_shared_and_foreign_slots_without_recomputing_gu(prepared_banks, kind, offset, rows, first, dtype):
    prepared, inputs, outputs, expected = baseline(prepared_banks, kind, offset, rows, first, dtype)
    before = outputs["activation"].clone()
    versions = [outputs[name]._version for name in ("local", "grouping", "out")]
    activation_version = outputs["activation"]._version
    prepared.run(*inputs, **outputs)
    torch.testing.assert_close(outputs["out"].view(torch.uint8), expected.view(torch.uint8), rtol=0, atol=0)
    assert torch.equal(outputs["activation"].view(torch.int16), before.view(torch.int16))
    assert outputs["activation"]._version == activation_version
    assert [outputs[name]._version for name in ("local", "grouping", "out")] == [v + 1 for v in versions]
    # A changed supplied activation MUST change routed down output. Recomputing
    # original GU would incorrectly restore1 and hide the new operator's work.
    outputs["activation"].reshape(rows, 11, 160)[:, :5].fill_(2.)
    prepared.run(*inputs, **outputs)
    torch.testing.assert_close(outputs["out"][:, :5], expected[:, :5] * 2, rtol=0, atol=0)
    torch.testing.assert_close(outputs["out"][:, 5:], expected[:, 5:], rtol=0, atol=0)


def test_poison_status_survives_all_foreign_and_shared_publication(prepared_banks):
    prepared, inputs, outputs, _ = baseline(prepared_banks, 42, 160, 1, 0)
    inputs[1][:, :10] = torch.arange(256, 266, dtype=torch.int32, device="cuda")
    outputs["activation"].reshape(1, 11, 160)[:, 10].fill_(float("nan"))
    prepared.run(*inputs, **outputs)
    assert torch.isnan(outputs["out"]).all()


@pytest.mark.parametrize("bad", ["negative-activation", "negative-output", "alias-local", "grad-activation"])
def test_native_readonly_activation_domain_refused_before_mutation(prepared_banks, bad):
    prepared, inputs, outputs, _ = baseline(prepared_banks, 20, 0, 1, 0)
    if bad == "negative-activation":
        outputs["activation"] = torch._neg_view(outputs["activation"])
    elif bad == "negative-output":
        outputs["out"] = torch._neg_view(outputs["out"])
    elif bad == "alias-local":
        outputs["local"] = outputs["activation"].view(torch.int64).reshape(-1)[:11].reshape(1, 11)
    else:
        outputs["activation"].requires_grad_(True)
    before = outputs["out"].clone()
    with pytest.raises(RuntimeError):
        prepared.run(*inputs, **outputs)
    torch.testing.assert_close(outputs["out"], before, rtol=0, atol=0)


def test_supplied_down_current_stream_and_capture(prepared_banks):
    prepared, inputs, outputs, expected = baseline(prepared_banks, 42, 480, 1, 256)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        prepared.run(*inputs, **outputs)
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        prepared.run(*inputs, **outputs)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(outputs["out"], expected, rtol=0, atol=0)

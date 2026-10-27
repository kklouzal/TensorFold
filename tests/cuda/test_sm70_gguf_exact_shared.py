"""The exact routed core initializes skipped shared slots before owner overwrite."""

import pytest

torch = pytest.importorskip("torch")

from test_sm70_gguf_exact import bits, exact_fixture  # noqa: E402
from tensorfold.cuda.sm70 import gguf_ops  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0),
                                reason="SM70 exact shared-slot ownership")


@pytest.mark.parametrize("kind", [8, 12, 13])
@pytest.mark.parametrize("width", [80, 160])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("rows", [1, 3])
def test_skipped_shared_slots_are_defined_zero_without_reading_shared_values(kind, width, dtype, index_dtype, rows):
    bank, _, _ = exact_fixture(kind, width=width)
    torch.manual_seed(1942 + rows)
    x = (torch.randn(rows, 2560, device="cuda") * .1).half()
    picks = torch.tensor([[2, 0, 1], [1, 2, 0], [0, 1, 2]], device="cuda", dtype=index_dtype)[:rows].contiguous()
    expected = bank.forward_reference(x, picks).to(dtype)
    for part in bank.shared:
        part.weight.fill_(float("nan"))
    output = torch.full_like(expected, float("nan"))
    gguf_ops.experts_exact(x, picks, bank.gate, bank.up, bank.down, bank.shared, output)
    shared = picks == 2
    bits(output[shared], torch.zeros_like(output[shared]))
    bits(output[~shared], expected[~shared])


def test_exact_shared_skip_tracks_live_routes_during_capture_and_preserves_invalid_nan_contract():
    bank, _, _ = exact_fixture(13)
    x = (torch.randn(3, 2560, device="cuda") * .1).half()
    picks = torch.tensor([[0, 1, 2], [1, 2, 0], [2, 0, 1]], device="cuda", dtype=torch.int64)
    output = torch.empty((3, 3, 2560), device="cuda")

    def run():
        gguf_ops.experts_exact(x, picks, bank.gate, bank.up, bank.down, bank.shared, output)

    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    picks.copy_(picks.roll(1, 1))
    picks[0] = torch.tensor([-1, 2, 2**40], device="cuda")
    graph.replay()
    shared = picks == 2
    bits(output[shared], torch.zeros_like(output[shared]))
    assert bool(torch.isnan(output[0, 0]).all()) and bool(torch.isnan(output[0, 2]).all())
    # The public owner still overwrites every shared slot with the original
    # Dense arithmetic, including arbitrary positions in the slot matrix.
    valid = torch.tensor([[2, 0, 1], [1, 2, 0], [0, 1, 2]], device="cuda")
    bits(bank.forward_exact(x, valid), bank.forward_reference(x, valid))

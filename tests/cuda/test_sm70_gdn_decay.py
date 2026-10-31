"""The two declared native decay representations agree where exp(A_log)=1 exactly."""

import pytest

torch = pytest.importorskip("torch")

from tensorfold.families.qwen4_exp.cuda import gdn  # noqa: E402 - optional backend guard precedes CUDA imports

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0),
                                reason="direct stored decay requires actual SM70 execution")


@pytest.mark.parametrize("nv", (6, 12, 24, 48))
def test_direct_decay_matches_exact_log_zero_control_in_capture_and_every_replay_prefix(nv):
    rows, nk = 3, nv // 3
    channels, width = gdn.widths(nk, nv)
    torch.manual_seed(903 + nv)
    projection = torch.randn(rows, width, device="cuda", dtype=torch.float16) * .1
    history = torch.randn(3, channels, device="cuda", dtype=torch.float16) * .1
    conv = torch.randn(channels, 4, device="cuda", dtype=torch.float16) * .1
    initial = torch.randn(nv, 128, 128, device="cuda") * .01
    bias = torch.randn(nv, device="cuda") * .1
    norm = torch.ones(128, dtype=torch.float16, device="cuda")
    log, negative = torch.zeros(nv, device="cuda"), -torch.ones(nv, device="cuda")
    live = torch.tensor([rows], dtype=torch.int32, device="cuda")
    buffers = []
    for _ in range(2):
        scratch = gdn.GDNScratch(rows, "cuda", nk, nv, dtype=torch.float16)
        scratch.keep_rows = live
        buffers.append((scratch, torch.empty_like(initial),
                        torch.empty(rows, nv * 128, dtype=torch.float16, device="cuda"),
                        torch.empty(rows, nv * 4, device="cuda")))

    def run(direct):
        scratch, final, output, sums = buffers[int(direct)]
        gdn.chain(projection, history, conv, initial, None if direct else log, bias, norm, 1e-6,
                  rows, scratch, final, output, sums, a_neg=negative if direct else None)

    run(False)
    run(True)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run(True)
    for kept in range(rows + 1):
        live.fill_(kept)
        run(False)
        graph.replay()
        torch.cuda.synchronize()
        for index in (1, 2, 3):
            torch.testing.assert_close(buffers[0][index], buffers[1][index], rtol=0, atol=0)
        for name in ("k", "v", "g", "b"):
            torch.testing.assert_close(getattr(buffers[0][0], name), getattr(buffers[1][0], name), rtol=0, atol=0)
        replayed = torch.empty_like(initial)
        gdn.replay(initial, buffers[1][0], kept, replayed)
        torch.testing.assert_close(replayed, buffers[1][1], rtol=0, atol=0)

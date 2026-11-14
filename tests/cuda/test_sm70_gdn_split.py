"""The multi-block prefill GDN recurrence is bitwise the serial chain kernel (outputs, sums, state, replay saves)."""

import pytest

torch = pytest.importorskip("torch")

from tensorfold.cuda.sm70 import gdn_native  # noqa: E402 - optional backend guard precedes CUDA imports
from tensorfold.families.qwen4_exp.cuda.gdn import GDNScratch, widths  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0),
                                reason="the FP16 GDN provider requires actual SM70 execution")


def weights(nv, negative, generator):
    nk = nv // 3
    channels, _ = widths(nk, nv)
    rate = torch.rand(nv, generator=generator, device="cuda") * 15 + 1
    return dict(nk=nk, nv=nv, negative=negative,
                conv=(torch.randn(channels, 4, generator=generator, device="cuda") * .4).half(),
                decay=(-rate if negative else rate.log()).contiguous(),
                bias=(torch.rand(nv, generator=generator, device="cuda") * 4 - 3).contiguous(),
                norm=(1 + .1 * torch.randn(128, generator=generator, device="cuda")).half())


def run(kind, w, projection, history, initial, rows, keep=None):
    nk, nv = w["nk"], w["nv"]
    scratch = GDNScratch(max(rows, 1), "cuda", nk, nv, dtype=torch.float16)
    for saved in (scratch.k, scratch.v, scratch.g, scratch.b):
        saved.fill_(float("nan"))
    live = None if keep is None else torch.tensor([keep], dtype=torch.int32, device="cuda")
    output = torch.full((max(rows, 1), nv * 128), float("nan"), dtype=torch.float16, device="cuda")
    sums = torch.full((max(rows, 1), nv * 4), float("nan"), device="cuda")
    final = torch.full_like(initial, float("nan"))
    getattr(gdn_native.extension(), kind)(projection, history, w["conv"], initial, w["decay"], w["bias"], w["norm"],
                                          1e-6, rows, output, sums, final, scratch.k, scratch.v, scratch.g,
                                          scratch.b, live, w["negative"])
    return [output[:rows], sums[:rows], final, scratch.k[:rows], scratch.v[:rows], scratch.g[:rows],
            scratch.b[:rows]]


def assert_bitwise(expected, actual):
    for name, a, b in zip(("output", "sums", "final", "k", "v", "g", "b"), expected, actual):
        assert a.shape == b.shape, name
        assert torch.equal(a.view(torch.int16 if a.dtype == torch.float16 else torch.int32),
                           b.view(torch.int16 if b.dtype == torch.float16 else torch.int32)), name


def inputs(w, rows, generator):
    channels, width = widths(w["nk"], w["nv"])
    projection = torch.randn(max(rows, 1), width, generator=generator, device="cuda").half()
    history = (torch.randn(3, channels, generator=generator, device="cuda") * .5).half()
    initial = torch.randn(w["nv"], 128, 128, generator=generator, device="cuda") * .1
    return projection, history, initial


@pytest.mark.parametrize("nv,rows", [(6, 1), (6, 3), (6, 63), (6, 64), (6, 1029), (12, 130), (24, 64), (48, 65)])
@pytest.mark.parametrize("negative", (True, False))
def test_split_chain_is_bitwise_the_serial_chain(nv, rows, negative):
    generator = torch.Generator(device="cuda").manual_seed(7000 + 97 * nv + rows + negative)
    w = weights(nv, negative, generator)
    projection, history, initial = inputs(w, rows, generator)
    assert_bitwise(run("chain", w, projection, history, initial, rows),
                   run("split_chain", w, projection, history, initial, rows))


@pytest.mark.parametrize("rows,keep", [(0, None), (64, 0), (1024, 700), (300, 300), (300, 5000), (300, -2)])
def test_split_chain_live_rows_match_the_serial_chain(rows, keep):
    generator = torch.Generator(device="cuda").manual_seed(7100 + rows)
    w = weights(6, True, generator)
    projection, history, initial = inputs(w, rows, generator)
    assert_bitwise(run("chain", w, projection, history, initial, rows, keep),
                   run("split_chain", w, projection, history, initial, rows, keep))


def test_split_chain_carries_state_across_chunks_like_the_serial_chain():
    generator = torch.Generator(device="cuda").manual_seed(7200)
    w = weights(6, True, generator)
    channels, _ = widths(w["nk"], w["nv"])
    projection, history, initial = inputs(w, 2048, generator)
    carried = {}
    for kind in ("chain", "split_chain"):
        conv, state, results = history, initial, []
        for chunk in projection.split(1024):
            result = run(kind, w, chunk.contiguous(), conv, state, 1024)
            conv = torch.cat([conv, chunk[:, :channels]])[-3:].contiguous()
            state = result[2]
            results.append(result)
        carried[kind] = results
    for expected, actual in zip(carried["chain"], carried["split_chain"]):
        assert_bitwise(expected, actual)


def test_dispatch_keeps_short_chains_and_captures_on_the_serial_kernel(monkeypatch):
    calls = []
    native = gdn_native.extension()

    class Spy:
        def __getattr__(self, name):
            calls.append(name)
            return getattr(native, name)

    monkeypatch.setattr(gdn_native, "extension", lambda: Spy())
    generator = torch.Generator(device="cuda").manual_seed(7300)
    w = weights(6, True, generator)
    rows = gdn_native.SPLIT_MIN_ROWS
    projection, history, initial = inputs(w, rows, generator)
    scratch = GDNScratch(rows, "cuda", 2, 6, dtype=torch.float16)
    final = torch.empty_like(initial)
    output = torch.empty(rows, 768, dtype=torch.float16, device="cuda")
    sums = torch.empty(rows, 24, device="cuda")

    def chain(count):
        gdn_native.chain(projection, history, w["conv"], initial, w["decay"], w["bias"], w["norm"], 1e-6, count,
                         scratch, final, output, sums, negative_decay=True)

    chain(rows - 1)
    chain(rows)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        chain(rows)
    assert calls == ["chain", "split_chain", "chain"]

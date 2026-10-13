"""Programmatic dependent launch of Flash Next's decode kernels (benchmark switch "pdl"): every PDL-launched kernel
waits for its producer before reading, so results equal the plain launches bit for bit, eager and in captured graphs,
under back-to-back producers and over many replays."""

import json
import os

import pytest
import torch

if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9:
    pytest.skip("PDL needs CUDA sm_90 or later", allow_module_level=True)

from test_flashnext_forward import V, _model  # noqa: E402
from test_flashnext_multi_graphs import PROMPTS, RQ, SAMPLINGS, _prefilled, _solo, _switch, _verify  # noqa: E402

from tensorfold.cuda import pdl  # noqa: E402
from tensorfold.cuda.kernels import qmm as shared  # noqa: E402
from tensorfold.cuda.streams import Stream  # noqa: E402
from tensorfold.families.qwen4_exp.cuda import qmm  # noqa: E402
from tensorfold.families.qwen4_exp.cuda.multi import MultiDecoder  # noqa: E402

# tests/cuda/conftest.py restores the process-wide launch policy (pdl.ON) after each test

def test_selection_is_per_chip_and_overridable(monkeypatch):
    monkeypatch.delenv(pdl.ENV, raising=False)
    assert pdl.select() == (torch.cuda.get_device_capability() == (12, 1))
    for value, want in (("1", True), ("0", False)):
        monkeypatch.setenv(pdl.ENV, value)
        assert pdl.select() is want
    monkeypatch.setenv(pdl.ENV, "yes")
    with pytest.raises(ValueError, match=pdl.ENV):
        pdl.select()


@pytest.mark.parametrize("n,k", [(2560, 6144), (13952, 2560), (4096, 1024), (320, 10240)])
@pytest.mark.parametrize("rows", [1, 3, 16, 17, 40])
def test_lane_matmul_reads_its_producers_rows_under_pdl(n, k, rows):
    """C1: the group-32 lane matmul (weights staged before the dependency wait, inputs after; clustered and staged
    K slices) launched with PDL right behind the kernel writing its input reads that kernel's rows: every output of
    a chain of 40 back-to-back producer/matmul pairs equals the plain launch's."""

    g = torch.Generator(device="cuda").manual_seed(n + rows)
    words = torch.randint(-(2**31), 2**31 - 1, (n, k // 8), generator=g, device="cuda", dtype=torch.int64) \
        .to(torch.int32)
    scales = (torch.rand((n, k // 32), generator=g, device="cuda") * 0.003 + 0.0003).to(torch.bfloat16)
    biases = (-scales.float() * 7.5).to(torch.bfloat16)
    q = qmm.make_q4(words, scales, biases)
    sources = [torch.randn((rows, k), generator=g, device="cuda").to(torch.bfloat16) for _ in range(40)]
    x = torch.empty((rows, k), dtype=torch.bfloat16, device="cuda")
    sk = qmm.split_for(q.n, q.k)
    part = torch.empty((max(sk, 1) * rows * q.n,), dtype=torch.float32, device="cuda")

    def chain(on: bool):
        pdl.ON = on
        outs = []
        for src in sources:
            torch.mul(src, 1, out=x)                     # a kernel producing x just before the matmul
            out = torch.empty((rows, q.n), dtype=torch.bfloat16, device="cuda")
            shared.matmul(x, q, qmm.group_sums(x), out=out, sk=sk, part=part)
            outs.append(out)
        torch.cuda.synchronize()
        return outs

    want = chain(False)
    for _ in range(3):
        got = chain(True)
        assert all(torch.equal(a, b) for a, b in zip(got, want))


@pytest.mark.parametrize("graphs", [False, True])
def test_decode_with_pdl_emits_each_streams_solo_tokens(monkeypatch, tmp_path, graphs):
    """C: rounds launched with PDL (eager, or replaying graphs captured with PDL edges) emit the solo runs' tokens,
    switching between "pdl" new and old every round (graphs: both variants captured, as an A/B run does)."""

    monkeypatch.setenv(pdl.ENV, "1")
    w = _model(7)
    refs = _solo(w, PROMPTS, SAMPLINGS, 20, RQ)
    _switch(monkeypatch, tmp_path)                   # an A/B run: ab_switch.active(), both PDL variants captured
    control = tmp_path / "ab.json"
    dec = MultiDecoder(w, slots=4, capacity=4096, depth=3, confidence=0.3, kv_dtype=RQ, prefill_rows=256,
                       graphs=graphs)
    assert dec.pdl
    if graphs:
        dec.capture_graphs()
        assert any(key[0] for key in dec.graphs.verify) and any(not key[0] for key in dec.graphs.verify)
    streams = [Stream(list(p), 20, smp, draft=i != 3) for i, (p, smp) in enumerate(zip(PROMPTS, SAMPLINGS))]
    for s in streams:
        dec.admit(s)
    rounds, replays = 0, 0
    while dec.live():
        control.write_text(json.dumps({"pdl": "old" if rounds % 2 else "new"}))
        os.utime(control, ns=(rounds * 10**9 + 7, rounds * 10**9 + 7))
        dec.finish(dec.round())
        replays += dec.replays
        rounds += 1
    assert [s.out for s in streams] == refs
    assert (replays > 0) == graphs


def test_replayed_pdl_graphs_keep_their_bits_over_many_replays(monkeypatch):
    """T15 (race stress): a verify forward captured with PDL edges, replayed 1,000 times over sparse and dense streams
    with the last round's rows folded first, writes the plain eager forward's logits, streams, caches and states every
    time."""

    monkeypatch.setenv(pdl.ENV, "1")
    w = _model(7)
    dec = MultiDecoder(w, slots=3, capacity=4096, depth=3, confidence=0.3, kv_dtype=RQ, graphs=True)
    dec.capture_graphs()
    assert all(key[0] for key in dec.graphs.verify)                  # no A/B run: the chip's PDL variant only
    g = torch.Generator().manual_seed(9)
    long = torch.randint(1, V, (2400,), generator=g).tolist()
    states = _prefilled(w, [long, PROMPTS[0], PROMPTS[1]], RQ, 4096)
    windows = [torch.randint(1, V, (k,), generator=g).tolist() for k in (4, 2, 3)]
    pdl.ON = True
    _verify(dec, w, states, windows, [[] for _ in states], 1, "eager")
    held = [[0, 1], [], [0]]
    pdl.ON = False
    want = _verify(dec, w, states, windows, held, 0, "eager")
    pdl.ON = True
    for i in range(1000):
        got = _verify(dec, w, states, windows, held, 0, "graph")
        assert all(torch.equal(a, b) for a, b in zip(got, want)), i

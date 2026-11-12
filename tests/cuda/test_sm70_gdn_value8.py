"""Unselected value8 native candidate against the unchanged native GDN oracle."""
from types import SimpleNamespace
import hashlib
from pathlib import Path
import json

import pytest
import torch

from tensorfold.cuda.sm70 import gdn_native, gdn_value8
from tensorfold.families.qwen4_exp.cuda.gdn import GDNScratch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0),
                                reason="requires actual SM70 component lease")


def exact(left, right):
    assert torch.equal(left.contiguous().view(torch.uint8), right.contiguous().view(torch.uint8))


@pytest.fixture(scope="module")
def candidate():
    return gdn_value8.prepare("cuda")


def setup(rows, pattern):
    generator = torch.Generator(device="cuda").manual_seed(1773 + rows)
    def half(shape):
        return torch.randn(shape, generator=generator, device="cuda", dtype=torch.float16) * .1
    p, history, weight = half((rows + 2, 2060)), half((3, 1280)), half((1280, 4))
    initial = torch.randn((6, 128, 128), generator=generator, device="cuda") * .01
    negative = -torch.tensor([.01, .1, 1, 3, 17, 127], device="cuda")
    bias = torch.randn((6,), generator=generator, device="cuda") * .1
    norm = half((128,)) + 1
    if pattern == "zero-key":
        p[:, :1280].zero_(); history.zero_(); weight.zero_()
    elif pattern == "extreme":
        p[:, 2048:2054] = torch.tensor([-30, 30, -1, 1, -0., 0.], device="cuda", dtype=torch.float16)
        p[:, 2054:2060] = torch.tensor([-1000, 1000, -20, 20, -1, 1], device="cuda", dtype=torch.float16)
        initial[..., ::2] *= -1
    elif pattern == "nonfinite":
        p[0, 0] = float("inf")
        p[0, 3] = float("nan")
    return p, history, weight, initial, negative, bias, norm


def outputs(rows, live):
    scratch = GDNScratch(rows + 2, "cuda", 2, 6, dtype=torch.float16)
    for name in ("k", "v", "g", "b"):
        getattr(scratch, name).fill_(-77)
    scratch.keep_rows = live
    return scratch, torch.empty((6, 128, 128), device="cuda"), \
        torch.full((rows + 2, 768), -31., device="cuda", dtype=torch.float16), \
        torch.full((rows + 2, 24), -17., device="cuda")


@pytest.mark.parametrize("rows", [1, 5, 32, 120])
@pytest.mark.parametrize("pattern", ["random", "zero-key", "extreme", "nonfinite"])
def test_coefficients_outputs_state_replay_and_inactive_rows_match_incumbent(candidate, rows, pattern, record_property):
    p, history, weight, initial, negative, bias, norm = setup(rows, pattern)
    live = torch.tensor([rows], device="cuda", dtype=torch.int32)
    reference, actual = outputs(rows, live), outputs(rows, live)
    workspace = gdn_value8.allocate_workspace(rows + 2, "cuda", device_budget_bytes=(rows + 2) * 1024)
    workspace.q.fill_(-91)
    prefixes = range(rows + 1) if rows <= 5 else sorted({0, 1, rows // 2, rows - 1, rows})
    for keep in prefixes:
        live.fill_(keep)
        old_scratch, old_final, old_out, old_sums = reference
        scratch, final, out, sums = actual
        gdn_native.chain(p, history, weight, initial, negative, bias, norm, 1e-6, rows,
                         old_scratch, old_final, old_out, old_sums, negative_decay=True)
        candidate.chain(p, history, weight, initial, negative, bias, norm, 1e-6, rows,
                        scratch, final, out, sums, workspace, negative_decay=True)
        exact(final, old_final); exact(out, old_out); exact(sums, old_sums)
        for name in ("k", "v", "g", "b"):
            exact(getattr(scratch, name), getattr(old_scratch, name))
        replayed = torch.empty_like(final)
        candidate.replay(initial, scratch, keep, replayed)
        exact(replayed, final)
        exact(workspace.q[rows:], torch.full_like(workspace.q[rows:], -91))
    record_property("value8_component", json.dumps({"rows": rows, "pattern": pattern,
        "candidate_module": candidate.module.__file__,
        "candidate_elf_sha256": hashlib.sha256(Path(candidate.module.__file__).read_bytes()).hexdigest(),
        "incumbent_module": gdn_native.extension().__file__,
        "incumbent_elf_sha256": hashlib.sha256(Path(gdn_native.extension().__file__).read_bytes()).hexdigest(),
        "qualification": "native component only; no model/P3 acceptance"}))


def test_captured_nondefault_stream_and_direct_decay_exact_log_zero_control(candidate):
    rows = 5
    p, history, weight, initial, _, bias, norm = setup(rows, "random")
    parameter = -torch.ones(6, device="cuda")
    live = torch.tensor([rows], device="cuda", dtype=torch.int32)
    scratch, final, out, sums = outputs(rows, live)
    workspace = gdn_value8.allocate_workspace(rows, "cuda", device_budget_bytes=rows * 1024)
    def run():
        candidate.chain(p, history, weight, initial, parameter, bias, norm, 1e-6, rows,
                        scratch, final, out, sums, workspace, negative_decay=True)
    run()
    graph, side = torch.cuda.CUDAGraph(), torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.graph(graph, stream=side):
        run()
    torch.cuda.current_stream().wait_stream(side)
    for keep in range(rows + 1):
        live.fill_(keep)
        graph.replay()
        expected = outputs(rows, live)
        gdn_native.chain(p, history, weight, initial, torch.zeros_like(parameter), bias, norm, 1e-6, rows,
                         *expected, negative_decay=False)
        for actual, old in zip((final, out, sums), expected[1:]):
            exact(actual, old)
    torch.cuda.synchronize()
    graph.reset()


def test_active_alias_rejected_before_any_buffer_mutation(candidate):
    rows = 1
    p, history, weight, initial, negative, bias, norm = setup(rows, "random")
    scratch, final, _, sums = outputs(rows, None)
    out = p.view(-1)[:3 * 768].view(3, 768)
    workspace = gdn_value8.allocate_workspace(rows, "cuda", device_budget_bytes=1024)
    before = p.clone()
    with pytest.raises(RuntimeError, match="aliases"):
        candidate.chain(p, history, weight, initial, negative, bias, norm, 1e-6, rows,
                        scratch, final, out, sums, workspace, negative_decay=True)
    exact(p, before)


def test_zero_rows_and_zero_prefix_copy_state_without_observing_empty_workspace(candidate):
    p, history, weight, initial, negative, bias, norm = setup(0, "random")
    scratch, final, out, sums = outputs(0, None)
    workspace = gdn_value8.allocate_workspace(0, "cuda", device_budget_bytes=0)
    candidate.chain(p, history, weight, initial, negative, bias, norm, 1e-6, 0,
                    scratch, final, out, sums, workspace, negative_decay=True)
    exact(final, initial)
    exact(out, torch.full_like(out, -31))
    exact(sums, torch.full_like(sums, -17))
    for name in ("k", "v", "g", "b"):
        exact(getattr(scratch, name), torch.full_like(getattr(scratch, name), -77))
    replayed = torch.empty_like(initial)
    candidate.replay(initial, scratch, 0, replayed)
    exact(replayed, initial)

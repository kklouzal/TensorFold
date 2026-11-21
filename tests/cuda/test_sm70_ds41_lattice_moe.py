"""DeepSeek-V4.1 prefill routed experts: the friend's TurboMind lattice path (ds41_lattice_moe) against the FP32
oracle and our fused WMMA path (ds41_prefill_moe), on real checkpoint bytes.

Asserted: finite, deterministic, probability-0 routes have no effect, no transcode overflow; vs the FP32 oracle the
exact lattice decode (default) within 5% of ours' rel L2 (same rounding points, different FP32 K order) and the
friend's FP16 decode (exact=False) within twice the 1Cat-rounding emulation. Printed (the A/B record): rel L2 of
the three paths vs the oracle, and per-layer time of the three on real 192-expert rank banks at 512..16384 rows.
"""

import pytest
import torch

from test_sm70_onecat_ds41 import LOCAL, Carriers, pytestmark  # noqa: F401
from test_sm70_ds41_prefill_moe import error, oracle, rank_banks, routes, timed

SIZES = (512, 2048, 8192)


@pytest.fixture(scope="module")
def ops():
    from tensorfold.cuda.sm70 import ds41_lattice_moe, ds41_prefill_moe

    ds41_prefill_moe.prepare()
    ds41_lattice_moe.prepare()
    return ds41_lattice_moe, ds41_prefill_moe, ds41_lattice_moe.Workspace(torch.device("cuda"))


@pytest.mark.parametrize("kind", [16, 17, 29])
def test_lattice_experts_against_fp32(ops, kind):
    lat, ours, ws = ops
    carriers = Carriers(kind)
    generator = torch.Generator().manual_seed(600 + kind)
    worst = [0.0, 0.0, 0.0]
    for r in (0, 1, 2, 3):
        banks, weights, left = carriers.rank(r)
        for m in SIZES:
            x = torch.randn(m, 5120, generator=generator).half().cuda()
            ids, p = routes(m, LOCAL, generator)
            expected = oracle(x, ids, p, weights)
            emulated = error(oracle(x, ids, p, weights, half_projections=True).half(), expected)
            out = lat.experts(x, ids, p, *banks, kind, kind, left, ws)
            assert torch.isfinite(out).all()
            again = lat.experts(x, ids, p, *banks, kind, kind, left, ws)
            assert torch.equal(out.view(torch.int16), again.view(torch.int16))
            friend = lat.experts(x, ids, p, *banks, kind, kind, left, ws, exact=False)
            assert torch.isfinite(friend).all()
            mine = error(ours.experts(x, ids, p, *banks, kind, kind, left), expected)
            got, fp16 = error(out, expected), error(friend, expected)
            worst = [max(worst[0], got[0]), max(worst[1], fp16[0]), max(worst[2], mine[0])]
            print(f"type {kind} rank {r} left {left} M{m}: lattice exact rel_l2 {got[0]:.4g} max_abs {got[1]:.3g} | "
                  f"lattice fp16 {fp16[0]:.4g} / {fp16[1]:.3g} | ours {mine[0]:.4g} / {mine[1]:.3g} | "
                  f"1Cat-rounding emulation {emulated[0]:.3g} / {emulated[1]:.3g} | ref_max "
                  f"{expected.abs().max().item():.3g}")
            # The exact path has ours' rounding points (only the FP32 K order differs).
            assert got[0] <= 1.05 * mine[0]
            assert fp16[0] <= 2 * emulated[0]
    ws.check()
    print(f"type {kind} worst rel_l2: lattice exact {worst[0]:.4g} lattice fp16 {worst[1]:.4g} ours {worst[2]:.4g}")


def test_lattice_dropped_routes(ops):
    lat, _, ws = ops
    banks, _, left = Carriers(29).rank(1)
    generator = torch.Generator().manual_seed(529)
    x = torch.randn(300, 5120, generator=generator).half().cuda()
    ids, p = routes(300, LOCAL, generator)
    base = lat.experts(x, ids, p, *banks, 29, 29, left, ws)
    moved = torch.where(p == 0, (ids + 3) % LOCAL, ids).to(torch.int32)
    assert torch.equal(base.view(torch.int16), lat.experts(x, moved, p, *banks, 29, 29, left, ws).view(torch.int16))
    none = lat.experts(x, ids, torch.zeros_like(p), *banks, 29, 29, left, ws)
    assert not none.view(torch.int16).any()


def layer_prefixes():
    from test_sm70_onecat_ds41 import checkpoint

    by_type = {}
    for name, t in sorted(checkpoint().tensors.items()):
        if name.endswith("ffn_down_exps.weight"):
            by_type.setdefault(t.ggml_type, name.removesuffix("ffn_down_exps.weight"))
    return [by_type[k] for k in sorted(by_type)]


def test_lattice_timing_192_experts(ops):
    lat, ours, ws = ops
    for prefix in layer_prefixes():
        prefix, banks, gate_type, down_type, left = rank_banks(prefix)
        generator = torch.Generator().manual_seed(41)
        for m in (512, 2048, 4096, 8192, 16384):
            x = torch.randn(m, 5120, generator=generator).half().cuda()
            ids = torch.rand(m, 384, generator=generator).argsort(-1)[:, :6]
            p = torch.softmax(torch.randn(m, 6, generator=generator), -1).float()
            mine = ids >= 192
            local = torch.where(mine, ids - 192, 0).to(torch.int32).cuda()
            p = torch.where(mine, p, 0).cuda()
            a = timed(lambda: lat.experts(x, local, p, *banks, gate_type, down_type, left, ws), 7)
            f = timed(lambda: lat.experts(x, local, p, *banks, gate_type, down_type, left, ws, exact=False), 7)
            b = timed(lambda: ours.experts(x, local, p, *banks, gate_type, down_type, left), 7)
            same = (lat.experts(x, local, p, *banks, gate_type, down_type, left, ws).float()
                    - ours.experts(x, local, p, *banks, gate_type, down_type, left).float())
            med = [t[len(t) // 2] for t in (a, f, b)]
            print(f"{prefix} types {gate_type}/{down_type} M{m}: lattice exact median {med[0]:.3f} ms "
                  f"[{a[0]:.3f}, {a[-1]:.3f}] | lattice fp16 median {med[1]:.3f} ms [{f[0]:.3f}, {f[-1]:.3f}] | "
                  f"ours median {med[2]:.3f} ms [{b[0]:.3f}, {b[-1]:.3f}] | ours/exact {med[2] / med[0]:.3f} "
                  f"fp16/exact {med[1] / med[0]:.3f} | exact-ours max |diff| {same.abs().max().item():.3g}")
    ws.check()
    print(f"lattice workspace {ws.nbytes / 2**20:.0f} MiB (tile {ws.tile})")

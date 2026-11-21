"""DeepSeek-V4.1 prefill routed experts (ds41_lattice_moe: the friend's TurboMind lattice path with our exact weight
decode) on real checkpoint bytes.

The oracle is an FP32 product of the NumPy-dequantized FP16 weights (test_sm70_onecat_ds41.reference, ggml-quants.c
order), with FP32 gate/up and the reference's FP16 SwiGLU output boundary. The prefill op is compared with it,
alongside the decode op (v41_experts per 6 rows, Q8_1 activations), an emulation of the 1Cat prefill rounding points
(FP16 gate/up and FP16 per-route down rows) and the friend's FP16 lattice arithmetic (exact=False). Asserted: the
exact path's error is no larger than the decode op's or the emulation's, outputs are deterministic, dropped
(probability 0) routes have no effect, no transcode overflow. Timing (printed, not asserted) runs real 192-expert
rank banks: both lattice arithmetics at 512..16384 rows and the exact path against the decode loop at the
MIN_TOKENS crossover.
"""

import time

import numpy as np
import pytest
import torch

from test_sm70_onecat_ds41 import LOCAL, Carriers, bank, pytestmark  # noqa: F401

SIZES = (65, 512, 2048, 8192)


@pytest.fixture(scope="module")
def ops():
    from tensorfold.cuda.sm70 import ds41_lattice_moe, onecat_ds41

    onecat_ds41.prepare()
    ds41_lattice_moe.prepare()
    return ds41_lattice_moe, onecat_ds41, ds41_lattice_moe.Workspace(torch.device("cuda"))


def routes(m, experts, generator, local_share=0.5):
    """Distinct top-6 ids per token; non-local routes get id 0 and probability 0 (Target.moe's form)."""
    ids = torch.rand(m, experts, generator=generator).argsort(-1)[:, :6].to(torch.int32)
    p = torch.softmax(torch.randn(m, 6, generator=generator), -1).float()
    mine = torch.rand(m, 6, generator=generator) < local_share
    return torch.where(mine, ids, 0).cuda(), torch.where(mine, p, 0).cuda()


def oracle(x, ids, p, weights, *, half_projections=False):
    """FP32 products of FP16 weights; FP16 down input; per-route FP16 down rows if half_projections."""
    gate, up, down = weights
    out = torch.zeros(x.shape, dtype=torch.float32, device="cuda")
    xf = x.float()
    for e in range(gate.shape[0]):
        token, slot = torch.nonzero((ids == e) & (p != 0), as_tuple=True)
        if not token.numel():
            continue
        rows = xf[token]
        g, u = rows @ gate[e].float().T, rows @ up[e].float().T
        if half_projections:
            g, u = g.half().float(), u.half().float()
        g, u = g.clamp(max=10), u.clamp(-10, 10)
        h = (g / (1 + torch.exp(-g)) * u * p[token, slot, None]).half().float()
        y = h @ down[e].float().T
        out.index_add_(0, token, y.half().float() if half_projections else y)
    return out


def decode_loop(dec, x, ids, p, banks, gate_type, down_type, left):
    out = torch.empty_like(x)
    for at in range(0, x.shape[0], 6):
        end = min(at + 6, x.shape[0])
        out[at:end] = dec.v41_experts(x[at:end].contiguous(), ids[at:end].contiguous(), p[at:end].contiguous(),
                                      *banks, gate_type, down_type, left)
    return out


def error(out, expected):
    diff = out.float() - expected
    return (diff.norm() / expected.norm()).item(), diff.abs().max().item()


@pytest.mark.parametrize("kind", [16, 17, 29])
def test_lattice_experts_against_fp32(ops, kind):
    lat, dec, ws = ops
    carriers = Carriers(kind)
    generator = torch.Generator().manual_seed(600 + kind)
    worst = [0.0, 0.0]
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
            got, fp16 = error(out, expected), error(friend, expected)
            worst = [max(worst[0], got[0]), max(worst[1], fp16[0])]
            line = (f"type {kind} rank {r} left {left} M{m}: lattice exact rel_l2 {got[0]:.4g} max_abs {got[1]:.3g} | "
                    f"lattice fp16 {fp16[0]:.4g} / {fp16[1]:.3g} | 1Cat-rounding emulation {emulated[0]:.3g} / "
                    f"{emulated[1]:.3g}")
            assert got[0] <= emulated[0]
            assert fp16[0] <= 2 * emulated[0]
            if m <= 2048:      # the decode loop is 6 rows per call
                decode = error(decode_loop(dec, x, ids, p, banks, kind, kind, left), expected)
                line += f" | decode {decode[0]:.4g} / {decode[1]:.3g}"
                assert got[0] <= decode[0]
            print(line + f" | ref_max {expected.abs().max().item():.3g}")
    ws.check()
    print(f"type {kind} worst rel_l2: lattice exact {worst[0]:.4g} lattice fp16 {worst[1]:.4g}")


@pytest.mark.parametrize("kind", [16, 29])
def test_lattice_dropped_routes(ops, kind):
    lat, _, ws = ops
    banks, _, left = Carriers(kind).rank(1)
    generator = torch.Generator().manual_seed(500 + kind)
    x = torch.randn(300, 5120, generator=generator).half().cuda()
    ids, p = routes(300, LOCAL, generator)
    base = lat.experts(x, ids, p, *banks, kind, kind, left, ws)
    moved = torch.where(p == 0, (ids + 3) % LOCAL, ids).to(torch.int32)
    assert torch.equal(base.view(torch.int16),
                       lat.experts(x, moved, p, *banks, kind, kind, left, ws).view(torch.int16))
    none = lat.experts(x, ids, torch.zeros_like(p), *banks, kind, kind, left, ws)
    assert not none.view(torch.int16).any()


def rank_banks(layer_prefix=None):
    """One real layer's 192-expert rank-1 banks (second EP2 half, ETP slice 1)."""
    from test_sm70_onecat_ds41 import checkpoint

    names = sorted(n for n in checkpoint().tensors if n.endswith("ffn_gate_exps.weight"))
    prefix = layer_prefix or names[len(names) // 2].removesuffix("ffn_gate_exps.weight")
    tensors = [checkpoint().tensors[prefix + f"ffn_{part}_exps.weight"] for part in ("gate", "up", "down")]
    kinds = [t.ggml_type for t in tensors]
    size = {16: 66, 17: 74, 29: 56}[kinds[2]]
    g, u, d = (bank(t)[192:384] for t in tensors)
    first = 576 // 256
    out = [torch.from_numpy(np.ascontiguousarray(a)).cuda()
           for a in (g[:, 576:1152], u[:, 576:1152], d[:, :, first * size:(first + 3) * size])]
    return prefix, out, kinds[0], kinds[2], 576 - 256 * first


def layer_prefixes():
    from test_sm70_onecat_ds41 import checkpoint

    by_type = {}
    for name, t in sorted(checkpoint().tensors.items()):
        if name.endswith("ffn_down_exps.weight"):
            by_type.setdefault(t.ggml_type, name.removesuffix("ffn_down_exps.weight"))
    return [by_type[k] for k in sorted(by_type)]


def timed(fn, repeats):
    fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1e3)
    return sorted(samples)


def rank_routes(m, generator):
    ids = torch.rand(m, 384, generator=generator).argsort(-1)[:, :6]
    p = torch.softmax(torch.randn(m, 6, generator=generator), -1).float()
    mine = ids >= 192
    return torch.where(mine, ids - 192, 0).to(torch.int32).cuda(), torch.where(mine, p, 0).cuda()


def test_lattice_timing_192_experts(ops):
    lat, dec, ws = ops
    for prefix in layer_prefixes():
        prefix, banks, gate_type, down_type, left = rank_banks(prefix)
        generator = torch.Generator().manual_seed(41)
        for m in (512, 2048, 4096, 8192, 16384):
            x = torch.randn(m, 5120, generator=generator).half().cuda()
            local, p = rank_routes(m, generator)
            a = timed(lambda: lat.experts(x, local, p, *banks, gate_type, down_type, left, ws), 7)
            f = timed(lambda: lat.experts(x, local, p, *banks, gate_type, down_type, left, ws, exact=False), 7)
            med = [t[len(t) // 2] for t in (a, f)]
            print(f"{prefix} types {gate_type}/{down_type} M{m}: lattice exact median {med[0]:.3f} ms "
                  f"[{a[0]:.3f}, {a[-1]:.3f}] | lattice fp16 median {med[1]:.3f} ms [{f[0]:.3f}, {f[-1]:.3f}] | "
                  f"fp16/exact {med[1] / med[0]:.3f}")
    ws.check()
    print(f"lattice workspace {ws.nbytes / 2**20:.0f} MiB (tile {ws.tile})")


def test_lattice_decode_crossover(ops):
    """The exact path against the decode op's per-6-row loop around MIN_TOKENS (Target.moe's switch)."""
    lat, dec, ws = ops
    prefix, banks, gate_type, down_type, left = rank_banks()
    generator = torch.Generator().manual_seed(43)
    for m in (65, 96, 128, 192, 256):
        x = torch.randn(m, 5120, generator=generator).half().cuda()
        local, p = rank_routes(m, generator)
        a = timed(lambda: lat.experts(x, local, p, *banks, gate_type, down_type, left, ws), 7)
        b = timed(lambda: decode_loop(dec, x, local, p, banks, gate_type, down_type, left), 5)
        print(f"{prefix} M{m} (MIN_TOKENS {lat.MIN_TOKENS}): lattice exact median {a[len(a) // 2]:.3f} ms | "
              f"decode loop median {b[len(b) // 2]:.3f} ms")

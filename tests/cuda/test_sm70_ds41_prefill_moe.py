"""DeepSeek-V4.1 prefill routed experts (grouped GEMMs with in-loop IQ decode) on real checkpoint bytes.

The oracle is an FP32 product of the NumPy-dequantized FP16 weights
(test_sm70_onecat_ds41.reference, ggml-quants.c order), with FP32 gate/up and
the reference's FP16 SwiGLU output boundary. The prefill op is compared with
it, alongside the decode op (v41_experts per 6 rows, Q8_1 activations) and an
emulation of the 1Cat prefill rounding points (FP16 gate/up and FP16 per-route
down rows). Asserted: the prefill error is no larger than either, outputs are
deterministic, dropped (probability 0) routes have no effect, and bad routed
ids raise. Timing (printed, not asserted) runs 192-expert rank banks.
"""

import time

import numpy as np
import pytest
import torch

from test_sm70_onecat_ds41 import LOCAL, Carriers, bank, pytestmark, tensor_of  # noqa: F401

SIZES = (7, 65, 512, 8192)


@pytest.fixture(scope="module")
def ops():
    from tensorfold.cuda.sm70 import ds41_prefill_moe, onecat_ds41

    onecat_ds41.prepare()
    ds41_prefill_moe.prepare()
    return ds41_prefill_moe, onecat_ds41


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


def decode_loop(dec, x, ids, p, banks, kind, left):
    out = torch.empty_like(x)
    for at in range(0, x.shape[0], 6):
        end = min(at + 6, x.shape[0])
        out[at:end] = dec.v41_experts(x[at:end].contiguous(), ids[at:end].contiguous(), p[at:end].contiguous(),
                                      *banks, kind, kind, left)
    return out


def error(out, expected):
    diff = out.float() - expected
    return (diff.norm() / expected.norm()).item(), diff.abs().max().item()


@pytest.mark.parametrize("kind", [16, 17, 29])
def test_prefill_experts_against_fp32(ops, kind):
    pre, dec = ops
    carriers = Carriers(kind)
    generator = torch.Generator().manual_seed(400 + kind)
    for r in (0, 3):
        banks, weights, left = carriers.rank(r)
        for m in SIZES:
            x = torch.randn(m, 5120, generator=generator).half().cuda()
            ids, p = routes(m, LOCAL, generator)
            expected = oracle(x, ids, p, weights)
            decode = error(decode_loop(dec, x, ids, p, banks, kind, left), expected)
            emulated = error(oracle(x, ids, p, weights, half_projections=True).half(), expected)
            out = pre.experts(x, ids, p, *banks, kind, kind, left)
            assert torch.isfinite(out).all()
            again = pre.experts(x, ids, p, *banks, kind, kind, left)
            assert torch.equal(out.view(torch.int16), again.view(torch.int16))
            got = error(out, expected)
            print(f"type {kind} rank {r} left {left} M{m}: prefill rel_l2 {got[0]:.3g} max_abs {got[1]:.3g} | "
                  f"decode {decode[0]:.3g} / {decode[1]:.3g} | 1Cat-rounding {emulated[0]:.3g} / "
                  f"{emulated[1]:.3g} | ref_max {expected.abs().max().item():.3g}")
            assert got[0] <= decode[0] and got[0] <= emulated[0]


@pytest.mark.parametrize("kind", [16, 29])
def test_dropped_routes_and_bad_ids(ops, kind):
    pre, _ = ops
    banks, _, left = Carriers(kind).rank(1)
    generator = torch.Generator().manual_seed(500 + kind)
    x = torch.randn(65, 5120, generator=generator).half().cuda()
    ids, p = routes(65, LOCAL, generator)
    base = pre.experts(x, ids, p, *banks, kind, kind, left)
    moved = torch.where(p == 0, (ids + 3) % LOCAL, ids).to(torch.int32)
    assert torch.equal(base.view(torch.int16),
                       pre.experts(x, moved, p, *banks, kind, kind, left).view(torch.int16))
    none = pre.experts(x, ids, torch.zeros_like(p), *banks, kind, kind, left)
    assert not none.view(torch.int16).any()
    bad = ids.clone()
    bad[3, int(torch.nonzero(p[3])[0])] = LOCAL
    with pytest.raises(RuntimeError, match="outside"):
        pre.experts(x, bad, p, *banks, kind, kind, left)


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


def test_prefill_timing_192_experts(ops):
    pre, dec = ops
    prefix, banks, gate_type, down_type, left = rank_banks()
    generator = torch.Generator().manual_seed(41)
    for m in (7, 65, 128, 256, 512, 2048, 8192):
        x = torch.randn(m, 5120, generator=generator).half().cuda()
        ids = torch.rand(m, 384, generator=generator).argsort(-1)[:, :6]
        p = torch.softmax(torch.randn(m, 6, generator=generator), -1).float()
        mine = ids >= 192
        local = torch.where(mine, ids - 192, 0).to(torch.int32).cuda()
        p = torch.where(mine, p, 0).cuda()
        a = timed(lambda: pre.experts(x, local, p, *banks, gate_type, down_type, left), 5)

        def loop():
            out = torch.empty_like(x)
            for at in range(0, m, 6):
                end = min(at + 6, m)
                out[at:end] = dec.v41_experts(x[at:end].contiguous(), local[at:end], p[at:end], *banks,
                                              gate_type, down_type, left)
            return out

        b = timed(loop, 3 if m >= 2048 else 5)
        print(f"{prefix} types {gate_type}/{down_type} M{m} routes {int((p != 0).sum())}: prefill median "
              f"{a[len(a) // 2]:.3f} ms [{a[0]:.3f}, {a[-1]:.3f}] | decode loop median {b[len(b) // 2]:.3f} ms "
              f"[{b[0]:.3f}, {b[-1]:.3f}] | workspace {pre.workspace_bytes(m, int((p != 0).sum())) / 2**20:.0f} MiB")

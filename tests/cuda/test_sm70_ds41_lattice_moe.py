"""DeepSeek-V4.1 prefill routed experts (ds41_lattice_moe: the friend's TurboMind lattice path with our exact weight
decode) on real checkpoint bytes.

The oracle is an FP32 product of the NumPy-dequantized FP16 weights (test_sm70_onecat_ds41.reference, ggml-quants.c
order), with FP32 gate/up and the reference's FP16 SwiGLU output boundary. The prefill op is compared with it,
alongside the decode op (v41_experts per 6 rows, Q8_1 activations) and an emulation of the 1Cat prefill rounding
points (FP16 gate/up and FP16 per-route down rows). Asserted: the exact path's error is no larger than the decode op's
or the emulation's, outputs are deterministic, dropped (probability 0) routes have no effect, no transcode overflow.
Timing (printed, not asserted) runs real 192-expert rank banks at 512..16384 rows and the exact path against the
decode loop at the MIN_TOKENS crossover.
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
    worst = 0.0
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
            got = error(out, expected)
            worst = max(worst, got[0])
            line = (f"type {kind} rank {r} left {left} M{m}: lattice exact rel_l2 {got[0]:.4g} max_abs {got[1]:.3g} | "
                    f"1Cat-rounding emulation {emulated[0]:.3g} / {emulated[1]:.3g}")
            assert got[0] <= emulated[0]
            if m <= 2048:      # the decode loop is 6 rows per call
                decode = error(decode_loop(dec, x, ids, p, banks, kind, kind, left), expected)
                line += f" | decode {decode[0]:.4g} / {decode[1]:.3g}"
                assert got[0] <= decode[0]
            print(line + f" | ref_max {expected.abs().max().item():.3g}")
    ws.check()
    print(f"type {kind} worst rel_l2: lattice exact {worst:.4g}")


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


@pytest.mark.parametrize("kind,paired", [(16, True), (17, True), (29, False)])
def test_transcode_uses_the_tensor_slice_expert_base(ops, kind, paired):
    lat, _, _ = ops
    (gate, up, down), _, down_offset = Carriers(kind).rank(1)
    source = gate if paired else down
    second = up if paired else None
    n, k, left = (1152, 5120, 0) if paired else (5120, 576, down_offset)
    _, wbytes, sbytes = lat.canonical(kind, n, k)
    extension = lat.extension()
    count = torch.ones(1, dtype=torch.int32, device="cuda")
    error = torch.zeros_like(count)

    def run(start, experts, selected):
        a = source[start:start + experts]
        b = second[start:start + experts] if second is not None else None
        index = torch.tensor([selected], dtype=torch.int32, device="cuda")
        w = torch.empty((experts, wbytes), dtype=torch.uint8, device="cuda")
        s = torch.empty((experts, sbytes), dtype=torch.uint8, device="cuda")
        extension.transcode(w, s, wbytes, sbytes, a, b, index, count, 80, kind, n, k, left, error)
        return w[selected], s[selected]

    direct = run(0, 3, 2)
    sliced = run(2, 1, 0)
    assert all(torch.equal(a, b) for a, b in zip(direct, sliced, strict=True))
    assert not error.item()


def test_transcode_rejects_noncovering_destination_and_unaligned_group(ops):
    lat, _, _ = ops
    _, wbytes, sbytes = lat.canonical(16, 32, 256)
    source = torch.zeros(2, 32, 66, dtype=torch.uint8, device="cuda")
    w = torch.empty(wbytes, dtype=torch.uint8, device="cuda")
    s = torch.empty(sbytes * 2, dtype=torch.uint8, device="cuda")
    ids = torch.tensor([1], dtype=torch.int32, device="cuda")
    count = torch.ones_like(ids)
    with pytest.raises(RuntimeError, match="incompatible destination banks"):
        lat.extension().transcode(w, s, wbytes, sbytes, source, None, ids, count, 80, 16, 32, 256, 0,
                                  torch.zeros_like(count))
    source = torch.zeros(2, 32, 132, dtype=torch.uint8, device="cuda")
    w = torch.empty(2 * wbytes, dtype=torch.uint8, device="cuda")
    with pytest.raises(RuntimeError, match="invalid packet group width or offset"):
        lat.extension().transcode(w, s, wbytes, sbytes, source, None, ids, count, 80, 16, 32, 256, 1,
                                  torch.zeros_like(count))


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
            print(f"{prefix} types {gate_type}/{down_type} M{m}: lattice exact median {a[len(a) // 2]:.3f} ms "
                  f"[{a[0]:.3f}, {a[-1]:.3f}]")
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


def _check_partitioned_lattice(ops, kind, etp):
    from test_sm70_onecat_ds41 import BYTES, reference, tensor_of

    lat, _, _ = ops
    tensors = [tensor_of(kind, part) for part in ("gate", "up", "down")]
    width, block = 2304 // etp, 256
    groups = 8 // etp
    local = 384 // groups
    selected = [0, 1, 2, 3, 4, 5, local - 2, local - 1]
    selected_ids = torch.tensor(selected, dtype=torch.int32, device="cuda")
    generator = torch.Generator().manual_seed(932 + kind)
    workspace = lat.Workspace("cuda", tile=local, intermediate=width)
    size = BYTES[kind]
    for shard in range(etp):
        # Exercise the tail EP groups and every intra-ETP input window.
        expert_start = (groups - etp + shard) * local
        first = shard * width
        first_block, left = divmod(first, block)
        blocks = -(-(left + width) // block)
        assert left == (0, 128)[shard] and blocks == (9 if etp == 1 else 5)
        g, u, d = (bank(t)[expert_start:expert_start + local] for t in tensors)
        raw = (g[:, first:first + width], u[:, first:first + width],
               d[:, :, first_block * size:(first_block + blocks) * size])
        banks = [torch.from_numpy(np.ascontiguousarray(a)).cuda() for a in raw]
        assert [a.shape[:2] for a in banks] == [(local, width), (local, width), (local, 5120)]
        assert banks[-1].shape[-1] == blocks * size
        decoded = []
        for index, item in enumerate(raw):
            active = np.ascontiguousarray(item[selected])
            values = reference(active.reshape(-1, active.shape[-1]), kind).reshape(len(selected), active.shape[1], -1)
            if index == 2:
                values = values[:, :, left:left + width]
            decoded.append(torch.from_numpy(values.astype(np.float16)).cuda())
        rows_cases = (128, 769, 2048) if etp == 1 else (128, 769)
        for rows in rows_cases:
            x = torch.randn(rows, 5120, generator=generator).half().cuda()
            ids, probabilities = routes(rows, len(selected), generator, local_share=0.25)
            local_ids = selected_ids[ids]
            expected = oracle(x, ids, probabilities, decoded)
            emulated = oracle(x, ids, probabilities, decoded, half_projections=True).half()
            output = lat.experts(x, local_ids, probabilities, *banks, kind, kind, left, workspace)
            repeated = lat.experts(x, local_ids, probabilities, *banks, kind, kind, left, workspace)
            assert torch.equal(output.view(torch.int16), repeated.view(torch.int16))
            assert torch.isfinite(output).all() and error(output, expected)[0] <= error(emulated, expected)[0]
            dropped = torch.where(probabilities == 0, local - 1, local_ids).contiguous()
            assert torch.equal(output.view(torch.int16), lat.experts(
                x, dropped, probabilities, *banks, kind, kind, left, workspace).view(torch.int16))
        if etp == 1:
            none = lat.experts(x[:128], local_ids[:128], torch.zeros_like(probabilities[:128]),
                               *banks, kind, kind, left, workspace)
            assert not none.view(torch.int16).any()
        del banks, decoded
    workspace.check()


@pytest.mark.parametrize("kind", [16, 17, 29])
def test_lattice_etp2_experts_against_fp32(ops, kind):
    _check_partitioned_lattice(ops, kind, 2)


@pytest.mark.parametrize("kind", [16, 17, 29])
def test_lattice_etp1_experts_against_fp32(ops, kind):
    _check_partitioned_lattice(ops, kind, 1)

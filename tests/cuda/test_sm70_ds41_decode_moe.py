"""DeepSeek-V4.1 expert-major decode MoE (ds41_decode_moe, the friend's patch 240 EM3) on real checkpoint bytes.

Contract checked: every output row equals the 6-row path's (onecat_ds41.v41_experts in pieces of 6) bit for
bit, for every gate/up and down variant, every TP slice (left 0/64/128/192), M 1..130 (pieces of 64), with
non-local routes passed as id 0 / probability 0 (Target.moe's form) and hot-expert routing; rows are batch
invariant; a captured CUDA graph replays new routing exactly. The FP32 oracle error (test_sm70_onecat_ds41's
reference) is reported, equal to the 6-row path's by construction.
"""

import pytest
import torch

from test_sm70_onecat_ds41 import LOCAL, Carriers, bits_equal, fp32_reference, pytestmark  # noqa: F401


@pytest.fixture(scope="module")
def ops():
    from tensorfold.cuda.sm70 import ds41_decode_moe, onecat_ds41

    onecat_ds41.prepare()
    return ds41_decode_moe, ds41_decode_moe.Workspace("cuda"), onecat_ds41


def pieces(dec, x, ids, p, banks, kind, left):
    out = torch.empty_like(x)
    for at in range(0, x.shape[0], 6):
        end = min(at + 6, x.shape[0])
        out[at:end] = dec.v41_experts(x[at:end].contiguous(), ids[at:end].contiguous(), p[at:end].contiguous(),
                                      *banks, kind, kind, left)
    return out


def routing(m, generator, *, hot=False, local_share=0.5):
    if hot:  # a few experts take most routes (concurrent / speculative rows)
        ids = torch.randint(0, 3, (m, 6), generator=generator)
        ids[:, 3:] = torch.randint(0, LOCAL, (m, 3), generator=generator)
    else:
        ids = torch.rand(m, LOCAL, generator=generator).argsort(-1)[:, :6]
    p = torch.softmax(torch.randn(m, 6, generator=generator), -1).float()
    mine = torch.rand(m, 6, generator=generator) < local_share
    return torch.where(mine, ids, 0).to(torch.int32).cuda(), torch.where(mine, p, 0).cuda()


def variant_table(em):
    table = em.extension().variants()
    return [v for v, row in enumerate(r for r in table if r[0] == 0)], \
        [v for v, row in enumerate(r for r in table if r[0] == 1)]


@pytest.mark.parametrize("kind", [16, 17, 29])
def test_schedule_bit_exact_against_pieces(ops, kind):
    em, ws, dec = ops
    carriers = Carriers(kind)
    generator = torch.Generator().manual_seed(700 + kind)
    for r in range(4):
        banks, weights, left = carriers.rank(r)
        for m in (*range(1, 34), 40, 48, 64, 65, 100, 130):
            x = torch.randn(m, 5120, generator=generator).half().cuda()
            ids, p = routing(m, generator, hot=m % 3 == 0)
            expected = pieces(dec, x, ids, p, banks, kind, left)
            out = ws.experts(x, ids, p, *banks, kind, kind, left)
            assert bits_equal(out, expected), (kind, r, m)
            if m in (1, 6, 32) and r in (0, 3):
                ref = fp32_reference(x, ids, p, weights)
                diff = out.float() - ref
                print(f"type {kind} rank {r} M{m}: rel_l2 vs FP32 {(diff.norm() / ref.norm()).item():.4g} "
                      f"max_abs {diff.abs().max().item():.4g} (bit-equal to the 6-row path)")
            # Batch invariance: row 0 alone equals row 0 of the batch.
            assert bits_equal(ws.experts(x[:1], ids[:1], p[:1], *banks, kind, kind, left), out[:1])


@pytest.mark.parametrize("kind", [16, 17, 29])
def test_every_variant_bit_exact(ops, kind):
    em, ws, dec = ops
    carriers = Carriers(kind)
    gate_up, down = variant_table(em)
    generator = torch.Generator().manual_seed(800 + kind)
    banks, _, left = carriers.rank(1)
    for m in (1, 7, 32, 64):
        x = torch.randn(m, 5120, generator=generator).half().cuda()
        ids, p = routing(m, generator, hot=m == 32)
        expected = pieces(dec, x, ids, p, banks, kind, left)
        for g in gate_up:
            schedule = {f"g{kind}": [(1, g)], f"d{kind}": [(1, 4)]}
            assert bits_equal(ws.experts(x, ids, p, *banks, kind, kind, left, schedule=schedule), expected), (m, g)
        for d in down:
            schedule = {f"g{kind}": [(1, 4)], f"d{kind}": [(1, d)]}
            assert bits_equal(ws.experts(x, ids, p, *banks, kind, kind, left, schedule=schedule), expected), (m, d)


def test_dropped_routes_and_all_dropped(ops):
    em, ws, dec = ops
    banks, _, left = Carriers(29).rank(2)
    generator = torch.Generator().manual_seed(29)
    x = torch.randn(9, 5120, generator=generator).half().cuda()
    ids, p = routing(9, generator)
    base = ws.experts(x, ids, p, *banks, 29, 29, left)
    moved = torch.where(p == 0, (ids + 3) % LOCAL, ids).to(torch.int32)
    assert bits_equal(base, ws.experts(x, moved, p, *banks, 29, 29, left))
    none = ws.experts(x, ids, torch.zeros_like(p), *banks, 29, 29, left)
    assert not none.view(torch.int16).any()  # +0 everywhere, like the 6-row path's
    assert bits_equal(none, pieces(dec, x, ids, torch.zeros_like(p), banks, 29, left))


def test_graph_replay_with_new_routing(ops):
    em, ws, dec = ops
    banks, _, left = Carriers(16).rank(0)
    generator = torch.Generator().manual_seed(16)
    m = 24
    x = torch.randn(m, 5120, generator=generator).half().cuda()
    ids, p = routing(m, generator)
    out = torch.empty_like(x)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        ws.experts(x, ids, p, *banks, 16, 16, left, out=out)  # warm (persistent grids query occupancy once)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        ws.experts(x, ids, p, *banks, 16, 16, left, out=out)
    for hot in (False, True, False):
        new_ids, new_p = routing(m, generator, hot=hot)
        x.copy_(torch.randn(m, 5120, generator=generator).half())
        ids.copy_(new_ids)
        p.copy_(new_p)
        graph.replay()
        torch.cuda.synchronize()
        assert bits_equal(out, pieces(dec, x, ids, p, banks, 16, left))

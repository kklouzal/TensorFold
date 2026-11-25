"""DeepSeek-V4.1 expert-major decode MoE (ds41_decode_moe, the friend's patch 240 EM3) on real checkpoint bytes.

Contract checked: every output row equals the 6-row path's (onecat_ds41.v41_experts in pieces of 6) bit for
bit, for every gate/up and down variant, every TP slice (left 0/64/128/192), M 1..130 (pieces of 64), with
non-local routes passed as id 0 / probability 0 (Target.moe's form) and hot-expert routing; rows are batch
invariant; a captured CUDA graph replays new routing exactly. The FP32 oracle error (test_sm70_onecat_ds41's
reference) is reported, equal to the 6-row path's by construction.
"""

import numpy as np
import pytest
import torch

from test_sm70_onecat_ds41 import BYTES, LOCAL, Carriers, bits_equal, fp32_reference, grid, pytestmark, tables  # noqa: F401


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


def etp2_banks(carriers, rank):
    start, width = rank * 1152, 1152
    size = BYTES[carriers.kind]
    first, left = divmod(start, 256)
    g, u, d = carriers.raw
    pieces = (g[:, start:start + width], u[:, start:start + width], d[:, :, first * size:(first + 5) * size])
    banks = [torch.from_numpy(np.ascontiguousarray(raw)).cuda() for raw in pieces]
    reference = [carriers.ref[0][:, start:start + width], carriers.ref[1][:, start:start + width],
                 carriers.ref[2][:, :, start:start + width]]
    return banks, reference, left


def etp2_iq2_xxs_reference(dec, x, ids, probabilities, banks, left, columns):
    """Independent scalar 36-group reduction at selected output columns."""
    q8 = dec.quantize_q8_1(x)
    gate, up, down = banks
    # The fixed six-row projection and quantizer operate on independent
    # groups. No ETP4 down multiply or intermediate output rounding is used.
    activations = []
    for start in (0, 576):
        projection = dec.v41_gate_up(q8, ids, gate[:, start:start + 576].contiguous(),
                                     up[:, start:start + 576].contiguous(), 16)
        activations.append(dec.v41_swiglu_q8(projection, probabilities))
    activation = torch.cat(activations, dim=-2).cpu().numpy()
    down = down[:, list(columns)].contiguous().cpu().numpy()
    picks, probs = ids.cpu().numpy(), probabilities.cpu().numpy()
    signs = tables()["ksigns_iq2xs"]
    codebook = grid("iq2xxs_grid").astype(np.int32)
    bits = 1 << np.arange(8)
    result = np.empty((len(picks), len(columns)), dtype=np.float16)
    for token, (expert_ids, weights) in enumerate(zip(picks, probs, strict=True)):
        for ci, _ in enumerate(columns):
            partials = np.zeros(6, dtype=np.float32)
            for route, (expert, probability) in enumerate(zip(expert_ids, weights, strict=True)):
                if probability == 0:
                    continue
                dot = np.float32(0)
                row = down[expert, ci].reshape(5, 66)
                for group, block in enumerate(activation[token, route]):
                    physical_block, lane = divmod(group + left // 32, 8)
                    raw = row[physical_block]
                    entries = raw[2 + lane * 8:6 + lane * 8]
                    metadata = int.from_bytes(raw[6 + lane * 8:10 + lane * 8], "little")
                    codes = np.concatenate([codebook[entry] * np.where(
                        int(signs[(metadata >> (7 * octet)) & 127]) & bits, -1, 1)
                        for octet, entry in enumerate(entries)])
                    integer_dot = int(np.dot(codes, block[4:].view(np.int8).astype(np.int32)))
                    d = np.float32(raw[:2].view("<f2")[0])
                    xd = np.float32(block[:2].view("<f2")[0])
                    scaled = (d * xd) * np.float32(integer_dot * (1 + 2 * (metadata >> 28)))
                    dot = dot + scaled * np.float32(0.125)
                partials[route] = np.float16(dot)
            total = np.float32(0)
            for warp in range(4):
                warp_total = np.float32(0)
                for route in range(warp, 6, 4):
                    if weights[route] != 0:
                        warp_total = warp_total + partials[route]
                total = total + warp_total
            result[token, ci] = np.float16(total)
    return torch.from_numpy(result).cuda()


@pytest.mark.parametrize("kind", [16, 17, 29])
def test_etp2_raw_expert_slices_and_127_rows(ops, kind):
    em, _, dec = ops
    ws = em.Workspace("cuda", intermediate=1152)
    carriers = Carriers(kind)
    generator = torch.Generator().manual_seed(1152 + kind)
    for rank in range(2):
        banks, reference, left = etp2_banks(carriers, rank)
        assert left == rank * 128
        x = torch.randn(127, 5120, generator=generator).half().cuda()
        ids, p = routing(127, generator, hot=True)
        ids[-1, 0] = LOCAL - 1  # the last expert and output row end exactly at the bank boundary
        p[-1, 0] = 0.25
        result = ws.experts(x, ids, p, *banks, kind, kind, left)
        assert torch.isfinite(result).all()
        for row in (0, 63, 64, 126):
            one = slice(row, row + 1)
            assert bits_equal(result[one], ws.experts(x[one], ids[one], p[one], *banks, kind, kind, left))
        # Preserve the maintained oracle's threshold for a gross defect.
        raw = fp32_reference(x[-1:], ids[-1:], p[-1:], reference)
        assert (result[-1:].float() - raw).norm() / raw.norm() < 0.1
        if kind == 16:
            columns = (0, 31, 5119)
            exact = etp2_iq2_xxs_reference(dec, x[-1:], ids[-1:], p[-1:], banks, left, columns)
            assert bits_equal(result[-1:, list(columns)], exact)
        with pytest.raises(RuntimeError, match="0/128"):
            ws.experts(x[:1], ids[:1], p[:1], *banks, kind, kind, 64)
        with pytest.raises(RuntimeError, match="matching workspace"):
            ws.experts(x[:1], ids[:1], p[:1], banks[0][:, :576].contiguous(),
                       banks[1][:, :576].contiguous(), banks[2], kind, kind, left)


def test_etp2_q8_group_workspace_replays_changed_routes(ops):
    em, default, _ = ops
    ws = em.Workspace("cuda", intermediate=1152)
    assert em.extension().workspace_bytes() == default.buffer.numel()
    assert default.buffer.numel() - ws.buffer.numel() == 635904
    banks, _, left = etp2_banks(Carriers(16), 1)
    generator = torch.Generator().manual_seed(112)
    x = torch.randn(7, 5120, generator=generator).half().cuda()
    ids, p = routing(7, generator)
    out = torch.empty_like(x)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        ws.experts(x, ids, p, *banks, 16, 16, left, out=out)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        ws.experts(x, ids, p, *banks, 16, 16, left, out=out)
    for hot in (True, False):
        next_ids, next_p = routing(7, generator, hot=hot)
        x.copy_(torch.randn(7, 5120, generator=generator).half())
        ids.copy_(next_ids)
        p.copy_(next_p)
        ids[-1] = -1
        p[-1] = -0.0
        graph.replay()
        eager = ws.experts(x, ids, p, *banks, 16, 16, left)
        assert bits_equal(out, eager) and not out[-1].view(torch.int16).any()

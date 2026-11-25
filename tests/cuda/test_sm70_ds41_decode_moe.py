"""DeepSeek-V4.1 expert-major decode MoE (ds41_decode_moe, the friend's patch 240 EM3) on real checkpoint bytes.

Contract checked: every output row equals the 6-row path's (onecat_ds41.v41_experts in pieces of 6) bit for
bit, for every gate/up and down variant, every TP slice (left 0/64/128/192), M 1..130 (pieces of 64), with
non-local routes passed as id 0 / probability 0 (Target.moe's form) and hot-expert routing; rows are batch
invariant; a captured CUDA graph replays new routing exactly. The wider ETP2 and ETP1 banks have their own
full-width FP32 oracle and an exact IQ2_XXS down reduction oracle at selected columns.
"""

import numpy as np
import pytest
import torch

from test_sm70_onecat_ds41 import (  # noqa: F401
    BYTES, LOCAL, Carriers, bank, bits_equal, fp32_reference, grid, pytestmark, reference, tables, tensor_of,
)


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


def routing(m, generator, *, hot=False, local_share=0.5, experts=LOCAL):
    if hot:  # a few experts take most routes (concurrent / speculative rows)
        ids = torch.randint(0, 3, (m, 6), generator=generator)
        ids[:, 3:] = torch.randint(0, experts, (m, 3), generator=generator)
    else:
        ids = torch.rand(m, experts, generator=generator).argsort(-1)[:, :6]
    p = torch.softmax(torch.randn(m, 6, generator=generator), -1).float()
    mine = torch.rand(m, 6, generator=generator) < local_share
    return torch.where(mine, ids, 0).to(torch.int32).cuda(), torch.where(mine, p, 0).cuda()


def variant_table(em):
    table = em.extension().variants()
    return [v for v, row in enumerate(r for r in table if r[0] == 0)], \
        [v for v, row in enumerate(r for r in table if r[0] == 1)]


def test_iq2_xxs_exact_final_bank_reads(ops):
    em, workspace, reference = ops
    generator = torch.Generator().manual_seed(1301)

    def bank(rows, blocks):
        raw = torch.randint(256, (1, rows, blocks, 66), dtype=torch.uint8, generator=generator)
        raw[..., 0] = 0
        raw[..., 1] = 0x14  # FP16 block scale 2**-10.
        return raw.flatten(-2).cuda()

    banks = bank(576, 20), bank(576, 20), bank(5120, 3)
    for carrier in banks:
        assert carrier.storage_offset() == 0
        assert carrier.untyped_storage().nbytes() == carrier.numel()
        assert (carrier.data_ptr() + carrier.numel() - 8) % 4 == 0
    x = torch.randn(1, 5120, generator=generator).half().cuda()
    ids = torch.zeros(1, 6, device="cuda", dtype=torch.int32)
    probabilities = torch.zeros(1, 6, device="cuda", dtype=torch.float32)
    probabilities[0, 0] = 1
    expected = pieces(reference, x, ids, probabilities, banks, 16, 192)
    assert torch.isfinite(expected).all()
    actual = workspace.experts(x, ids, probabilities, *banks, 16, 16, 192)
    assert bits_equal(actual, expected)
    gate, up, down = banks
    compact = pack_down_fragments(down, 16, 192, False)
    actual = workspace.experts(x, ids, probabilities, gate, up, compact, 16, 16, 192, fragment_down=True)
    assert bits_equal(actual, expected)
    padded = pack_down_fragments(down, 16, 192, True)
    actual.fill_(float("nan"))
    fragments(em, workspace, actual, x, ids, probabilities, (gate, up, padded), 16, 192, True)
    assert bits_equal(actual, expected)


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


def wide_iq2_xxs_reference(dec, x, ids, probabilities, banks, left, columns):
    """Independent scalar full-width reduction at selected output columns."""
    q8 = dec.quantize_q8_1(x)
    gate, up, down = banks
    # The fixed six-row projection and quantizer operate on independent
    # groups. No ETP4 down multiply or intermediate output rounding is used.
    activations = []
    for start in range(0, gate.shape[1], 576):
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
                row = down[expert, ci].reshape(-1, 66)
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
            exact = wide_iq2_xxs_reference(dec, x[-1:], ids[-1:], p[-1:], banks, left, columns)
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
    assert_wide_graph_replay(ws, banks, left, torch.Generator().manual_seed(112))


def assert_wide_graph_replay(ws, banks, left, generator):
    expert_count = banks[0].shape[0]
    x = torch.randn(7, 5120, generator=generator).half().cuda()
    ids, p = routing(7, generator, experts=expert_count)
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
        next_ids, next_p = routing(7, generator, hot=hot, experts=expert_count)
        next_ids[0, 0] = expert_count - 1
        next_p[0, 0] = 0.25
        x.copy_(torch.randn(7, 5120, generator=generator).half())
        ids.copy_(next_ids)
        p.copy_(next_p)
        ids[-1] = -1
        p[-1] = -0.0
        graph.replay()
        eager = ws.experts(x, ids, p, *banks, 16, 16, left)
        assert bits_equal(out, eager) and not out[-1].view(torch.int16).any()


@pytest.mark.parametrize("kind", [16, 17, 29])
def test_etp1_raw_expert_banks(ops, kind):
    em, default, dec = ops
    ws = em.Workspace("cuda", intermediate=2304)
    assert default.buffer.numel() - ws.buffer.numel() == 138240
    raw = [bank(tensor_of(kind, part))[-48:] for part in ("gate", "up", "down")]
    size = BYTES[kind]
    assert [a.shape for a in raw] == [(48, 2304, 20 * size), (48, 2304, 20 * size), (48, 5120, 9 * size)]
    banks = [torch.from_numpy(np.array(a, copy=True, order="C")).cuda() for a in raw]

    generator = torch.Generator().manual_seed(2304 + kind)
    x = torch.randn(127, 5120, generator=generator).half().cuda()
    ids, p = routing(127, generator, hot=True, local_share=0.125, experts=48)
    ids[-2].fill_(-1)
    p[-2].fill_(-0.0)
    active = (0, 20, 47)
    ids[-1].copy_(torch.tensor([*active, 0, 0, 0], dtype=torch.int32))
    p[-1].copy_(torch.tensor([0.25, 0.25, 0.25, 0, 0, 0], dtype=torch.float32))
    result = ws.experts(x, ids, p, *banks, kind, kind, 0)
    assert torch.isfinite(result).all() and not result[-2].view(torch.int16).any()
    for r in (0, 63, 64, 126):
        row = slice(r, r + 1)
        assert bits_equal(result[row], ws.experts(x[row], ids[row], p[row], *banks, kind, kind, 0))

    weights = []
    for a in raw:
        chosen = a[list(active)]
        dequantized = reference(chosen.reshape(-1, a.shape[-1]), kind).astype(np.float16)
        weights.append(torch.from_numpy(dequantized).cuda().view(3, a.shape[1], -1))
    mapped = torch.tensor([[0, 1, 2, 0, 0, 0]], dtype=torch.int32, device="cuda")
    full = fp32_reference(x[-1:], mapped, p[-1:], weights)
    assert (result[-1:].float() - full).norm() / full.norm() < 0.1
    with pytest.raises(RuntimeError, match="must be 0 for ETP1"):
        ws.experts(x[:1], ids[:1], p[:1], *banks, kind, kind, 128)
    if kind == 16:
        columns = (0, 31, 5119)
        exact = wide_iq2_xxs_reference(dec, x[-1:], ids[-1:], p[-1:], banks, 0, columns)
        assert bits_equal(result[-1:, list(columns)], exact)
        short = banks[2][:1, :, :-size].contiguous()
        with pytest.raises(RuntimeError, match="matching workspace and left"):
            ws.experts(x[:1], ids[:1], p[:1], banks[0][:1], banks[1][:1], short, kind, kind, 0)
        assert_wide_graph_replay(ws, banks, 0, generator)


def pack_down_fragments(down, kind, left, padded):
    """Independent test packer, outside the measured operator."""
    blocks = down.reshape(*down.shape[:-1], 3, down.shape[-1] // 3)
    low, high = [], []
    for ordinal in range(left // 32, left // 32 + 18):
        block, sub = divmod(ordinal, 8)
        row = blocks[:, :, block]
        if kind == 29:
            low.append(row[:, :, 4 * sub:4 * sub + 4])
            high.append(row[:, :, 32 + 2 * sub:34 + 2 * sub])
        else:
            low.append(row[:, :, 2 + 8 * sub:10 + 8 * sub])
            if kind == 17:
                high.append(row[:, :, 66 + sub:67 + sub])
    if kind == 29:
        scales = blocks[..., 48:56].flatten(-2)
        fields = [scales, *low, *high]
    else:
        scales = blocks[..., :2].flatten(-2)
        fields = [*low, scales, *high]
    if padded:
        count = -sum(field.shape[-1] for field in fields) % 16
        fields.append(torch.full((*down.shape[:2], count), 0xA5, dtype=down.dtype, device=down.device))
    return torch.cat(fields, dim=-1)


@pytest.mark.parametrize("kind,size", [(16, 66), (17, 74), (29, 56)])
def test_host_down_fragment_export_matches_independent_packer(ops, kind, size):
    em, _, _ = ops
    ext = em.extension()
    generator = torch.Generator().manual_seed(1433 + kind)
    raw = torch.randint(256, (2, 5, 9, size), dtype=torch.uint8, generator=generator)
    stride = ext.compact_down_row_bytes(kind)
    owner = torch.full((raw.numel() + 10 * stride + 2,), 0xA5, dtype=torch.uint8)
    source = owner[1:1 + raw.numel()]
    destination = owner[1 + raw.numel():-1]
    source.copy_(raw.flatten())
    for cut in range(4):
        first_block, left = divmod(18 * cut, 8)
        window = raw[:, :, first_block:first_block + 3].contiguous().flatten(-2)
        expected = pack_down_fragments(window, kind, left * 32, False)
        ext.pack_down_fragments(source, destination, kind, 18 * cut)
        assert torch.equal(destination.view_as(expected), expected)
        assert torch.equal(source, raw.flatten())
        assert owner[0].item() == owner[-1].item() == 0xA5
    with pytest.raises(RuntimeError, match="complete source rows"):
        ext.pack_down_fragments(source[:-1], destination, kind, 0)
    with pytest.raises(RuntimeError, match="legal rank cut"):
        ext.pack_down_fragments(source, destination, kind, 1)
    with pytest.raises(RuntimeError, match="disjoint"):
        ext.pack_down_fragments(source, source[:destination.numel()], kind, 0)


def fragments(em, ws, out, x, ids, probabilities, banks, kind, left, padded):
    m = x.shape[0]
    em.extension().experts_fragments(
        ws.buffer, out, x, ids, probabilities, *banks, kind, kind, left,
        em.pick("g", kind, m), em.pick("d", kind, m), padded,
    )


@pytest.mark.parametrize("kind,rows", [(16, (1, 2, 32)), (17, (1, 6, 8, 16, 24)), (29, (1, 32))])
def test_down_fragments_bit_exact(ops, kind, rows):
    em, ws, dec = ops
    carriers = Carriers(kind)
    generator = torch.Generator().manual_seed(900 + kind)
    for rank in range(4):
        banks, _, left = carriers.rank(rank)
        gate, up, down = banks
        alternatives = [(padded, pack_down_fragments(down, kind, left, padded)) for padded in (False, True)]
        for m in rows:
            x = torch.randn(m, 5120, generator=generator).half().cuda()
            ids, p = routing(m, generator, hot=m > 1)
            # Always exercise the last row of the final bank.
            ids[0, 0], p[0, 0] = LOCAL - 1, 0.25
            expected = pieces(dec, x, ids, p, banks, kind, left)
            assert bits_equal(ws.experts(x, ids, p, *banks, kind, kind, left), expected)
            for padded, fragment in alternatives:
                result = torch.full_like(x, float("nan"))
                fragments(em, ws, result, x, ids, p, (gate, up, fragment), kind, left, padded)
                assert bits_equal(result, expected), (kind, left, m, padded)


@pytest.mark.parametrize("kind", [16, 17, 29])
def test_down_fragments_captured_replay(ops, kind):
    em, ws, dec = ops
    banks, _, left = Carriers(kind).rank(3)
    gate, up, down = banks
    generator = torch.Generator().manual_seed(1000 + kind)
    m = 8
    x = torch.randn(m, 5120, generator=generator).half().cuda()
    ids, p = routing(m, generator)
    for padded in (False, True):
        fragment_banks = gate, up, pack_down_fragments(down, kind, left, padded)
        result = torch.empty_like(x)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            fragments(em, ws, result, x, ids, p, fragment_banks, kind, left, padded)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            fragments(em, ws, result, x, ids, p, fragment_banks, kind, left, padded)
        for hot, all_dropped in ((True, False), (False, True)):
            new_ids, new_p = routing(m, generator, hot=hot)
            x.copy_(torch.randn(m, 5120, generator=generator).half())
            ids.copy_(new_ids)
            p.copy_(new_p)
            if all_dropped:
                p.zero_()
            graph.replay()
            torch.cuda.synchronize()
            assert bits_equal(result, pieces(dec, x, ids, p, banks, kind, left))


def test_down_fragments_reject_bad_component_inputs(ops):
    em, ws, _ = ops
    (gate, up, down), _, left = Carriers(29).rank(0)
    fragment = pack_down_fragments(down, 29, left, False)
    x = torch.zeros(1, 5120, device="cuda", dtype=torch.float16)
    ids, p = routing(1, torch.Generator().manual_seed(1300))
    result = torch.empty_like(x)
    with pytest.raises(RuntimeError, match="row width or alignment"):
        fragments(em, ws, result, x, ids, p, (gate, up, fragment[..., :-2].contiguous()), 29, left, False)
    with pytest.raises(RuntimeError, match="row width or alignment"):
        fragments(em, ws, result, x, ids, p, (gate, up, fragment), 29, left, True)
    unaligned = torch.empty(fragment.numel() + 1, dtype=torch.uint8, device="cuda")[1:].view_as(fragment)
    with pytest.raises(RuntimeError, match="row width or alignment"):
        fragments(em, ws, result, x, ids, p, (gate, up, unaligned), 29, left, False)
    with pytest.raises(RuntimeError, match="left offset"):
        fragments(em, ws, result, x, ids, p, (gate, up, fragment), 29, 32, False)
    with pytest.raises(RuntimeError, match="scheduled ordinary variant"):
        em.extension().experts_fragments(
            ws.buffer, result, x, ids, p, gate, up, fragment, 29, 29, left, em.pick("g", 29, 1), 0, False,
        )
    for width in (1152, 2304):
        wide = em.Workspace("cuda", intermediate=width)
        gate_wide = torch.empty((1, width, gate.shape[2]), dtype=torch.uint8, device="cuda")
        with pytest.raises(RuntimeError, match="fragments require ETP4"):
            em.extension().experts_fragments(
                wide.buffer, result, x, ids, p, gate_wide, gate_wide, fragment[:1], 29, 29, 0, 4, 4, False,
            )

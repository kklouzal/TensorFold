"""Packed (patch 87) MU kernels vs the vendored upstream single-sequence packed code (ported from the friend
overlay snapshot lp-snap-c1es8-numa-admit2-imgtext-share2-fix5-ram4-stepbud-ram3-memfix-ctx1m-ppovl-opt2-pfx-1-eng-pfxfix4,
tests/ds41mu_packed_check.py; imports adapted, sections, oracles and tolerances verbatim).

    CPU: TRITON_INTERPRET=1 CUDA_VISIBLE_DEVICES= python mu_packed_check.py [section ...]
    GPU: DS41MU_DEVICE=cuda python mu_packed_check.py [section ...]
"""

import random
import sys
import types

import torch
import torch.nn.functional as F  # noqa: F401
from mu_testlib import DEVICE, qdq

from tensorfold.families.deepseek_v4.cuda.kernels import (
    attention_ops,
    compression,
    engram,
    index_topk,
    kv_pack,
    mu_packed,
    mu_paged,
    mu_reindex,
    mu_slots,
    mu_state,
    mu_topk,
    quant_rope,
)

_PAYLOAD = {
    "local_mu_state": mu_state,
    "local_mu_paged": mu_paged,
    "local_mu_packed": mu_packed,
    "local_mu_model": mu_slots,
    "local_ds41_mu_topk": mu_topk,
    "local_ds41_mu_reindex": mu_reindex,
}
_UPSTREAM = {
    "vllm/model_executor/layers/deepseek_v41_engram.py": engram,
    "vllm/model_executor/kernels/deepseek_v41_compression.py": compression,
    "vllm/models/deepseek_v41/index_topk.py": index_topk,
}


def load_payload(name):
    return _PAYLOAD[name]


def load_upstream(path):
    return _UPSTREAM[path]


def load_attention_ops():
    return attention_ops


def install_packed_modules():
    return kv_pack, attention_ops, index_topk


def load_quant_rope():
    return quant_rope

torch.manual_seed(0)
random.seed(0)
SECTIONS = set(sys.argv[1:])
kv, ops, topk_mod = install_packed_modules()
qr = load_quant_rope()
st = load_payload("local_mu_state")
pg = load_payload("local_mu_paged")
pk = load_payload("local_mu_packed")
fp16_ops = load_attention_ops()
BS = 16


def section(name):
    return not SECTIONS or name in SECTIONS


def make_batch(lengths, starts, pad_tokens=0):
    qsl = [0]
    positions = []
    for n, s in zip(lengths, starts):
        positions += list(range(s, s + n))
        qsl.append(qsl[-1] + n)
    total = qsl[-1]
    positions += [0] * pad_tokens
    valid = torch.zeros(total + pad_tokens, dtype=torch.bool)
    valid[:total] = True
    qsl = torch.tensor(qsl, dtype=torch.int32)
    n = total + pad_tokens
    token_req = st.build_token_req(qsl, len(lengths), n, torch.empty(n, dtype=torch.int32))
    return qsl, torch.tensor(positions, dtype=torch.int64), valid, token_req


def tables(totals, pages, width=48):
    pool = list(range(1, pages))
    random.shuffle(pool)
    table = torch.full((len(totals), width), -1, dtype=torch.int32)
    for r, total in enumerate(totals):
        for b in range((total + BS - 1) // BS):
            table[r, b] = pool.pop()
    return table


def page_of(table, r, position):
    return int(table[r, position // BS])


LAYOUTS = [pk.PackedLayout({2: 2, 8: 2, 14: 2}, BS), pk.PackedLayout({20: 1}, BS)]
assert LAYOUTS[0].bytes_per_token == 534 and LAYOUTS[1].bytes_per_token == 356

# ------------------------------------------------------------- lossless
if section("lossless"):
    for dim, group, e4m3, comp in ((512, 16, True, True), (128, 32, False, False)):
        x = (torch.randn(37, dim) * torch.rand(37, 1) * 4).half()
        x[3] = 0  # all-zero group: minimum scale
        x[5, :group] = 1e-6
        rows = kv.pack_rows(x, comp)
        back = kv.unpack_rows(rows, comp)
        ref = qdq(qr, x, fp4=True, group=group, e4m3_scale=e4m3)
        assert torch.equal(back.view(torch.int16), ref.view(torch.int16)), ("lossless", dim)
    print("lossless OK")


def upstream_rows(values, positions, valid, ratio, compressed, rows):
    """Upstream _insert_packed into position-addressed rows (one request)."""
    dim, group, size = (512, 16, 288) if compressed else (128, 32, 68)
    ops._insert_packed[(values.shape[0], dim // group)](
        values.contiguous(), positions.contiguous(), valid.contiguous(), rows, 0,
        RATIO=ratio, DIM=dim, GROUP=group, ROW=size, COMPRESSED=compressed,
    )
    return rows


# ------------------------------------------------------------- insert / rows
if section("insert") or section("rows") or section("prefill"):
    for trial in range(6):
        layout = random.choice(LAYOUTS)
        layer = random.choice(sorted(layout.regions))
        ratio, rows_per_page, c_off, i_off = layout.regions[layer]
        r = random.randint(1, 4)
        starts = [random.randint(0, 120) for _ in range(r)]
        lengths = [random.choice([1, 1, 2, 3, 7, 12]) for _ in range(r)]
        pad = random.randint(0, 2)
        qsl, positions, valid, token_req = make_batch(lengths, starts, pad)
        totals = [s + n for s, n in zip(starts, lengths)]
        table = tables(totals, 40)
        n = positions.numel()
        for compressed in (True, False):
            dim, size, off = (512, 288, c_off) if compressed else (128, 68, i_off)
            values = (torch.randn(n, dim) * 2).half()
            cache = torch.randint(0, 256, (40, layout.page_bytes), dtype=torch.uint8)
            expect = cache.clone()
            for rr in range(r):
                s0, s1 = int(qsl[rr]), int(qsl[rr + 1])
                rows = torch.zeros(totals[rr] // ratio + 1, size, dtype=torch.uint8)
                upstream_rows(values[s0:s1], positions[s0:s1], valid[s0:s1], ratio, compressed, rows)
                for p in range(starts[rr], totals[rr]):
                    if (p + 1) % ratio:
                        continue
                    page = page_of(table, rr, p)
                    at = off + ((p % BS) // ratio) * size
                    expect[page, at : at + size] = rows[p // ratio]
            pk.insert_rows(
                cache, values, positions, valid, token_req, table, layout=layout,
                layer=layer, compressed=compressed,
            )
            assert torch.equal(cache, expect), ("insert", trial, compressed)
            # dummy batch: token_req -1 writes nothing
            before = cache.clone()
            pk.insert_rows(
                cache, values, positions, valid, torch.full_like(token_req, -1), table,
                layout=layout, layer=layer, compressed=compressed,
            )
            assert torch.equal(cache, before), "dummy batch wrote packed rows"
            # rows: position-addressed copy of each request
            for rr in range(r):
                end = totals[rr] // ratio
                got = pk.gather_request_rows(
                    cache, table[rr], end, layout=layout, layer=layer, compressed=compressed
                )
                for c in range(end):
                    p = c * ratio + ratio - 1
                    page = page_of(table, rr, p)
                    at = off + (c % rows_per_page) * size
                    assert torch.equal(got[c], cache[page, at : at + size]), ("rows", trial, c)
            if not compressed:
                # prefill keys: paged unpack == upstream _unpack_index_keys of the rows
                for rr in range(r):
                    end = totals[rr] // ratio
                    rows = pk.gather_request_rows(
                        cache, table[rr], end, layout=layout, layer=layer, compressed=False
                    )
                    width = max(end, 64)
                    got = pk.unpack_request_index(cache, table[rr], end, width, layout=layout, layer=layer)
                    ref = torch.empty(width, 128, dtype=torch.float16)
                    pos = torch.tensor([totals[rr] - 1], dtype=torch.int64)
                    padded = torch.zeros(width, 68, dtype=torch.uint8)
                    padded[: rows.shape[0]] = rows
                    ops._unpack_index_keys[((width + 63) // 64,)](
                        padded, pos, ref, width, 1, RATIO=ratio, BN=64
                    )
                    assert torch.equal(got.view(torch.int16), ref.view(torch.int16)), ("prefill keys", trial, rr)
    for name in ("insert", "rows", "prefill"):
        if section(name):
            print(f"{name} OK")


# ------------------------------------------------------------- gather + ring
if section("gather") or section("ring"):
    S = 6
    for trial in range(5):
        layout = LAYOUTS[0]
        layer = 8
        ratio, rows_per_page, c_off, _ = layout.regions[layer]
        r = random.randint(1, 4)
        slots = random.sample(range(S), r)
        starts = [random.choice([0, random.randint(1, 300)]) for _ in range(r)]
        lengths = [random.choice([1, 1, 2, 5, 9]) for _ in range(r)]
        pad = random.randint(0, 2)
        qsl, positions, valid, token_req = make_batch(lengths, starts, pad)
        totals = [s + n for s, n in zip(starts, lengths)]
        n = positions.numel()
        req_slot = torch.full((16,), S, dtype=torch.int32)
        req_slot[:r] = torch.tensor(slots, dtype=torch.int32)
        # window history of every request: positions < start live in the ring
        # (and, for the FP16 reference, in a paged window cache)
        ring = (torch.randn(S + 1, pk.RING, 512)).half()
        wtable = tables(totals, 60)
        wcache = torch.randn(60, BS, 512).half()
        history = {}
        for rr in range(r):
            for p in range(max(0, starts[rr] - pk.RING), starts[rr]):
                v = torch.randn(512).half()
                history[(rr, p)] = v
                ring[slots[rr], p % pk.RING] = v
                wcache[page_of(wtable, rr, p), p % BS] = v
        window = torch.randn(n, 512).half()
        # FP16 reference window cache also holds this forward's latents
        for t in range(int(qsl[-1])):
            rr = int(token_req[t])
            p = int(positions[t])
            wcache[page_of(wtable, rr, p), p % BS] = window[t]
        # packed compressed rows of every request
        cache = torch.zeros(40, layout.page_bytes, dtype=torch.uint8)
        ctable = tables(totals, 40)
        lat_rows = {}
        for rr in range(r):
            end = totals[rr] // ratio
            vals = (torch.randn(end, 512) * 2).half()
            prow = kv.pack_rows(vals, True)
            lat_rows[rr] = prow
            for c in range(end):
                p = c * ratio + ratio - 1
                at = c_off + (c % rows_per_page) * 288
                cache[page_of(ctable, rr, p), at : at + 288] = prow[c]
        K = 8
        topk = torch.full((n, K), -1, dtype=torch.int32)
        for t in range(n):
            live = (int(positions[t]) + 1) // ratio
            pick = random.sample(range(max(live, 1)), min(K, live)) if live else []
            topk[t, : len(pick)] = torch.tensor(sorted(pick), dtype=torch.int32)
        if section("gather"):
            for first, last in ((0, n), (0, max(1, n // 2)), (max(1, n // 2), n)):
                if last <= first:
                    continue
                keys = torch.empty(last - first, 128 + K, 512).half()
                unused = torch.empty(last - first, 128 + K, dtype=torch.bool)
                pk.gather_attention(
                    keys, unused, first, last, positions, valid, token_req, qsl, req_slot,
                    ring, window, cache=cache, block_table=ctable, layout=layout,
                    layer=layer, topk=topk,
                )
                # window part == patch 85's FP16 paged gather (same latents)
                k85 = torch.empty(last - first, 128 + K, 512).half()
                u85 = torch.empty(last - first, 128 + K, dtype=torch.bool)
                ccache16 = torch.zeros(40, BS // ratio, 512).half()
                pg.mu_gather_attention(
                    k85, u85, positions[first:last], valid[first:last], token_req[first:last],
                    wcache, wtable, window_block=BS, compressed_cache=ccache16,
                    compressed_table=ctable, compressed_block=BS, ratio=ratio,
                    topk=topk[first:last],
                )
                assert torch.equal(unused[:, :128], u85[:, :128]), ("window mask", trial)
                assert torch.equal(keys[:, :128], k85[:, :128]), ("window keys", trial)
                assert torch.equal(unused[:, 128:], u85[:, 128:]), ("compressed mask", trial)
                for t in range(first, last):
                    rr = int(token_req[t])
                    for j in range(K):
                        c = int(topk[t, j])
                        got = keys[t - first, 128 + j]
                        if rr < 0 or c < 0:
                            assert not got.any() and bool(unused[t - first, 128 + j])
                            continue
                        ref = kv.unpack_rows(lat_rows[rr][c : c + 1].contiguous(), True)[0]
                        assert torch.equal(got.view(torch.int16), ref.view(torch.int16)), ("compressed key", trial, t, j)
        if section("ring"):
            before = ring.clone()
            pk.insert_ring(ring, window, positions, valid, token_req, qsl, req_slot)
            expect = before.clone()
            for rr in range(r):
                s0, s1 = int(qsl[rr]), int(qsl[rr + 1])
                last_p = int(positions[s1 - 1])
                for t in range(s0, s1):
                    p = int(positions[t])
                    if p > last_p - pk.RING:
                        expect[slots[rr], p % pk.RING] = window[t]
            assert torch.equal(ring, expect), ("ring", trial)
            assert torch.equal(ring[S], before[S]), "scratch ring written"
    for name in ("gather", "ring"):
        if section(name):
            print(f"{name} OK")


# ------------------------------------------------------------- decode indexer
if section("index"):
  exact = worst = 0
  checks = 0
  fp32_rounding_seen = False
  # TensorFold: index heads per rank 16 (overlay), 8 (TP4: 32 / 4) and 4 (TP8: 32 / 8).
  for H in (16, 8, 4):
    for trial in range(4):
        layout = random.choice(LAYOUTS)
        layer = random.choice(sorted(layout.regions))
        ratio, rows_per_page, _, i_off = layout.regions[layer]
        r = random.randint(1, 3)
        starts = [random.randint(1, 200) for _ in range(r)]
        lengths = [1] * r
        qsl, positions, valid, token_req = make_batch(lengths, starts, 1)
        totals = [s + 1 for s in starts]
        n = positions.numel()
        table = tables(totals, 40)
        cache = torch.zeros(40, layout.page_bytes, dtype=torch.uint8)
        pos_rows = {}
        for rr in range(r):
            end = totals[rr] // ratio
            vals = (torch.randn(max(end, 1), 128) * 0.5).half()
            prow = kv.pack_rows(vals, False)
            pos_rows[rr] = prow
            for c in range(end):
                p = c * ratio + ratio - 1
                at = i_off + (c % rows_per_page) * 68
                cache[page_of(table, rr, p), at : at + 68] = prow[c]
        q = qdq(qr, (torch.randn(n, H, 128) * 0.5).half(), fp4=True, group=32)
        w = (torch.randn(n, H) * 0.1).half()
        width = 512 // ratio
        out = torch.full((n, width), float("nan")).half()
        pk.index_scores(q, w, cache, table, positions, valid, token_req, layout=layout, layer=layer, out=out)
        # TensorFold TP8: an FP32 out holds the unrounded head sums; rounding them once gives the FP16 store.
        out32 = torch.full((n, width), float("nan"))
        pk.index_scores(q, w, cache, table, positions, valid, token_req, layout=layout, layer=layer, out=out32)
        for rr in range(r):
            t = int(qsl[rr])
            end = totals[rr] // ratio
            assert torch.equal(out32[t, :end].half().view(torch.int16), out[t, :end].view(torch.int16)), (
                "fp32 out", H, trial, rr)
            fp32_rounding_seen |= bool((out32[t, :end] != out32[t, :end].half().float()).any())
        for rr in range(r):
            t = int(qsl[rr])
            end = totals[rr] // ratio
            keys = torch.zeros(width + 1, 68, dtype=torch.uint8)
            keys[: pos_rows[rr].shape[0]] = pos_rows[rr]
            qp = torch.zeros(64, 128).half()
            qp[:H] = q[t]
            wp = torch.zeros(64).half()
            wp[:H] = w[t]
            ref = torch.full((1, keys.shape[0]), float("nan")).half()
            ops._index_scores_decode[((keys.shape[0] + 255) // 256,)](
                qp, wp, keys, positions[t : t + 1].contiguous(), ref, 1, ref.shape[1],
                RATIO=ratio, H=H, R=64, BN=64, BLOCKS=4, num_warps=4,
            )
            a, b = out[t, :end].float(), ref[0, :end].float()
            checks += 1
            exact += int(torch.equal(out[t, :end].view(torch.int16), ref[0, :end].view(torch.int16)))
            if end:
                worst = max(worst, (a - b).abs().max().item())
                assert torch.allclose(a, b, rtol=2e-3, atol=2e-3), ("index scores", trial, rr)
        # isolation: other requests' pages and queries replaced
        target = random.randrange(r)
        t = int(qsl[target])
        cache2 = torch.randint(0, 256, cache.shape, dtype=torch.uint8)
        used = table[target][table[target] >= 0].long()
        cache2[used] = cache[used]
        q2 = q.clone()
        others = [i for i in range(n) if i != t]
        q2[others] = qdq(qr, (torch.randn(len(others), H, 128)).half(), fp4=True, group=32)
        out2 = torch.full((n, width), float("nan")).half()
        pk.index_scores(q2, w, cache2, table, positions, valid, token_req, layout=layout, layer=layer, out=out2)
        end = totals[target] // ratio
        assert torch.equal(out2[t, :end].view(torch.int16), out[t, :end].view(torch.int16)), "contamination"
  assert fp32_rounding_seen, "FP32 index scores were never finer than FP16"
  print(f"index OK ({exact}/{checks} rows bit-identical to upstream, max diff {worst:.2e})")


# ------------------------------------------------------------- loop selector
if section("select"):
    H = 16
    layout = LAYOUTS[0]
    layer = 2
    ratio, rows_per_page, _, i_off = layout.regions[layer]
    for lengths, starts in (([1, 1], [700, 1300]), ([1, 3], [1500, 1100])):
        r = len(lengths)
        qsl, positions, valid, token_req = make_batch(lengths, starts, 0)
        totals = [s + n for s, n in zip(starts, lengths)]
        table = tables(totals, 300, width=128)
        cache = torch.zeros(300, layout.page_bytes, dtype=torch.uint8)
        for rr in range(r):
            end = totals[rr] // ratio
            prow = kv.pack_rows((torch.randn(end, 128) * 0.5).half(), False)
            for c in range(end):
                p = c * ratio + ratio - 1
                at = i_off + (c % rows_per_page) * 68
                cache[page_of(table, rr, p), at : at + 68] = prow[c]
        n = positions.numel()
        q = qdq(qr, (torch.randn(n, H, 128) * 0.5).half(), fp4=True, group=32)
        w = (torch.randn(n, H) * 0.1).half()
        for rr in range(r):
            s0, s1 = int(qsl[rr]), int(qsl[rr + 1])
            end = totals[rr] // ratio
            rows = pk.gather_request_rows(cache, table[rr], end, layout=layout, layer=layer, compressed=False)
            got = ops.select_packed_positions(
                q[s0:s1].contiguous(), w[s0:s1].contiguous(), rows, positions[s0:s1],
                valid[s0:s1], ratio, lambda x: x,
            )
            # the single-sequence store: position-addressed rows of the full table
            full = torch.zeros(4096, 68, dtype=torch.uint8)
            full[:end] = rows[:end]
            ref = ops.select_packed_positions(
                q[s0:s1].contiguous(), w[s0:s1].contiguous(), full, positions[s0:s1],
                valid[s0:s1], ratio, lambda x: x,
            )
            assert torch.equal(got, ref), ("loop selector", lengths, rr)
    print("select OK")

# ------------------------------------------------------------- attend (TensorFold)
if section("attend") and DEVICE == "cpu":
    # attention_ops._attend_fused uses torch.bmm(..., out_dtype=torch.float32), a CUDA-only overload.
    print("attend SKIPPED (CPU: torch.bmm out_dtype is CUDA-only; run with DS41MU_DEVICE=cuda)")
elif section("attend"):
    # mu_packed.attend (gather_attention + attention_ops._attend_fused per pass) at attention heads per rank
    # 16 (TP4) and 8 (TP8), vs an FP32 torch restatement over the same gathered keys (FP16 PV boundary).
    S = 4
    layout = LAYOUTS[0]
    layer = 8
    ratio, rows_per_page, c_off, _ = layout.regions[layer]
    for heads in (16, 8):
        r = 3
        slots = [2, 0, 3]
        starts = [0, 140, 37]
        lengths = [5, 1, 3]
        qsl, positions, valid, token_req = make_batch(lengths, starts, 1)
        totals = [s + n for s, n in zip(starts, lengths)]
        n = positions.numel()
        req_slot = torch.full((16,), S, dtype=torch.int32)
        req_slot[:r] = torch.tensor(slots, dtype=torch.int32)
        ring = (torch.randn(S + 1, pk.RING, 512) * 0.3).half()
        window = (torch.randn(n, 512) * 0.3).half()
        cache = torch.zeros(40, layout.page_bytes, dtype=torch.uint8)
        ctable = tables(totals, 40)
        for rr in range(r):
            end = totals[rr] // ratio
            prow = kv.pack_rows((torch.randn(max(end, 1), 512) * 0.3).half(), True)
            for c in range(end):
                p = c * ratio + ratio - 1
                at = c_off + (c % rows_per_page) * 288
                cache[page_of(ctable, rr, p), at : at + 288] = prow[c]
        K = 8
        topk = torch.full((n, K), -1, dtype=torch.int32)
        for t in range(n):
            live = (int(positions[t]) + 1) // ratio
            pick = random.sample(range(max(live, 1)), min(K, live)) if live else []
            topk[t, : len(pick)] = torch.tensor(sorted(pick), dtype=torch.int32)
        query = (torch.randn(n, heads, 512) * 0.3).half()
        sink = torch.randn(heads)
        step = types.SimpleNamespace(valid=valid, token_req=token_req, qsl=qsl)
        got = pk.attend(query, positions, step, req_slot, ring, window, sink, cache=cache, block_table=ctable,
                        layout=layout, source=layer, topk=topk, tokens_per_pass=4)
        keys = torch.empty(n, 128 + K, 512).half()
        unused = torch.empty(n, 128 + K, dtype=torch.bool)
        pk.gather_attention(keys, unused, 0, n, positions, valid, token_req, qsl, req_slot, ring, window,
                            cache=cache, block_table=ctable, layout=layout, layer=layer, topk=topk)
        s = torch.einsum("thd,tkd->thk", query.float(), keys.float()) * 512**-0.5
        s = s.masked_fill(unused[:, None, :], float("-inf"))
        m = torch.maximum(s.amax(-1), sink[None, :])
        e = torch.exp(s - m[..., None])
        probs = (e / (e.sum(-1) + torch.exp(sink[None, :] - m))[..., None]).half()
        ref = torch.einsum("thk,tkd->thd", probs.float(), keys.float())
        assert got.shape == (n, heads, 512) and got.dtype == torch.float16
        assert torch.allclose(got.float(), ref, rtol=2e-2, atol=2e-3), ("attend", heads, (got.float() - ref).abs().max())
    print("attend OK")

print("ALL OK")

"""Patches 170/171 (hierarchical indexer, segmented top-k) checks (ported from the friend overlay snapshot
lp-snap-c1es8-numa-admit2-imgtext-share2-fix5-ram4-stepbud-ram3-memfix-ctx1m-ppovl-opt2-pfx-1-eng-pfxfix4, tests/ds41mu_reindex_check.py).
Imports adapted; the select section calls the vLLM-free mu_reindex.select / mu_packed.select_positions
with the same stub indexer query and weights the overlay computed inside its selectors. Oracles and
tolerances verbatim.

    CPU: TRITON_INTERPRET=1 CUDA_VISIBLE_DEVICES= python mu_reindex_check.py [section ...]
    GPU: DS41MU_DEVICE=cuda python mu_reindex_check.py [section ...]
"""
# ruff: noqa: B023  -- the overlay's per-iteration helpers are called within the iteration they close over.

import os
import random
import sys

import torch
import torch.nn.functional as F
from mu_testlib import DEVICE, qdq  # noqa: F401

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
import types

torch.manual_seed(0)
random.seed(0)
SECTIONS = set(sys.argv[1:])
kv, ops, topk_mod = install_packed_modules()
qr = load_quant_rope()
st = load_payload("local_mu_state")
pk = load_payload("local_mu_packed")
tk = load_payload("local_ds41_mu_topk")
rx = load_payload("local_ds41_mu_reindex")
BS = 16
# index heads per TP rank: 8 = index_n_heads 32 / TP4 (overlay); TensorFold also runs 4 (TP8) via
# DS41MU_INDEX_HEADS=4 (test_mu_interpret runs the scores and select sections that way too).
H = int(os.environ.get("DS41MU_INDEX_HEADS", "8"))


def section(name):
    return not SECTIONS or name in SECTIONS


def upstream_candidate_blocks(logits, compress_lens, topk_blocks, block_size):
    """inference/model.py select_candidate_blocks (rev 1bae0e10), ties -> lowest block."""
    width = logits.size(-1)
    scores = F.pad(logits, (0, -width % block_size), value=-torch.inf)
    scores = scores.unflatten(-1, (-1, block_size)).amax(dim=-1)
    num_blocks = scores.size(-1)
    last = (compress_lens - 1) // block_size
    scores = scores.masked_fill(torch.arange(num_blocks) == last, torch.inf)
    k = min(topk_blocks, num_blocks)
    order = scores.float().sort(dim=-1, descending=True, stable=True)
    keep = torch.zeros_like(scores, dtype=torch.bool).scatter_(
        -1, order.indices[..., :k], order.values[..., :k] > -torch.inf
    )
    return keep.repeat_interleave(block_size, dim=-1)[..., :width]


def make_batch(lengths, starts):
    qsl = [0]
    positions = []
    for n, s in zip(lengths, starts):
        positions += list(range(s, s + n))
        qsl.append(qsl[-1] + n)
    total = qsl[-1]
    valid = torch.ones(total, dtype=torch.bool)
    qsl_t = torch.tensor(qsl, dtype=torch.int32)
    token_req = st.build_token_req(qsl_t, len(lengths), total, torch.empty(total, dtype=torch.int32))
    return qsl_t, torch.tensor(positions, dtype=torch.int64), valid, token_req


def build_pages(layout, layer, totals, width=200):
    ratio, rows_per_page, _, i_off = layout.regions[layer]
    pages = sum((t + BS - 1) // BS for t in totals) + 4
    pool = list(range(1, pages))
    random.shuffle(pool)
    table = torch.full((len(totals), width), -1, dtype=torch.int32)
    cache = torch.randint(0, 256, (pages, layout.page_bytes), dtype=torch.uint8)
    rows = {}
    for r, total in enumerate(totals):
        for b in range((total + BS - 1) // BS):
            table[r, b] = pool.pop()
        end = total // ratio
        # Clustered magnitudes so block maxima differ (and some ties exist).
        vals = (torch.randn(max(end, 1), 128) * (0.2 + torch.rand(max(end, 1), 1))).half()
        prow = kv.pack_rows(vals, False)
        rows[r] = prow
        for c in range(end):
            p = c * ratio + ratio - 1
            at = i_off + (c % rows_per_page) * 68
            cache[int(table[r, p // BS]), at : at + 68] = prow[c]
    return cache, table, rows


def queries(n):
    q = qdq(qr, (torch.randn(n, H, 128) * 0.5).half(), fp4=True, group=32)
    w = (torch.randn(n, H) * 0.1).half()
    return q, w


# ------------------------------------------------------------------ topkseg
if section("topkseg"):
    for width, rows in ((1000, 3), (16384, 2), (20000, 2), (40000, 2)):
        scores = (torch.randn(rows, width) * 3).half()
        scores[:, ::7] = scores[:, :1]  # ties at the threshold
        positions = torch.tensor([width - 1 - 37 * i for i in range(rows)], dtype=torch.int64)
        valid = torch.ones(rows, dtype=torch.bool)
        if rows > 2:
            valid[1] = False
        a = torch.full((rows, 512), -9, dtype=torch.int32)
        b = torch.full((rows, 512), -9, dtype=torch.int32)
        topk_mod.index_topk(a, scores, positions, valid, 1, 512)
        tk.topk(b, scores, positions, valid, 1, 512)
        assert torch.equal(a, b), ("topkseg", width)
    print("topkseg OK")

LAYOUT = pk.PackedLayout({20: 1}, BS)

# ------------------------------------------------------------------ scores
if section("scores"):
    for trial in range(3):
        totals = [random.randint(40, 700) for _ in range(3)]
        cache, table, _ = build_pages(LAYOUT, 20, totals)
        qsl, positions, valid, token_req = make_batch([1, 1, 1], [t - 1 for t in totals])
        q, w = queries(3)
        width = 768
        full = torch.full((3, width), float("nan")).half()
        pk.index_scores(q, w, cache, table, positions, valid, token_req, layout=LAYOUT, layer=20, out=full)
        blocks = 6
        cand = torch.full((3, blocks), -1, dtype=torch.int32)
        for t in range(3):
            nb = (totals[t] + 7) // 8
            pick = sorted(random.sample(range(nb - 1), min(blocks - 1, nb - 1))) + [nb - 1]
            cand[t, : len(pick)] = torch.tensor(pick, dtype=torch.int32)
        out = torch.full((3, blocks * 8), float("nan")).half()
        rx.candidate_scores(q, w, cache, table, positions, valid, token_req, cand,
                            layout=LAYOUT, layer=20, block=8, out=out)
        # TensorFold TP8: FP32 out = unrounded head sums; rounding once gives the FP16 store, dead slots unset.
        out32 = torch.full((3, blocks * 8), float("nan"))
        rx.candidate_scores(q, w, cache, table, positions, valid, token_req, cand,
                            layout=LAYOUT, layer=20, block=8, out=out32)
        live32 = ~torch.isnan(out)
        assert torch.equal(torch.isnan(out32), torch.isnan(out)), "fp32 dead slots"
        assert torch.equal(out32[live32].half().view(torch.int16), out[live32].view(torch.int16)), "fp32 scores"
        for t in range(3):
            n = totals[t]
            for j in range(blocks * 8):
                b = int(cand[t, j // 8])
                p = b * 8 + j % 8
                if b >= 0 and p < n:
                    assert out[t, j].view(torch.int16) == full[t, p].view(torch.int16), ("score", t, j)
                else:
                    assert torch.isnan(out[t, j]), ("dead slot written", t, j)
        # isolation: other requests' pages / queries replaced
        cache2 = torch.randint(0, 256, cache.shape, dtype=torch.uint8)
        used = table[1][table[1] >= 0].long()
        cache2[used] = cache[used]
        q2 = q.clone()
        q2[[0, 2]] = queries(2)[0]
        out2 = torch.full_like(out, float("nan"))
        rx.candidate_scores(q2, w, cache2, table, positions, valid, token_req, cand,
                            layout=LAYOUT, layer=20, block=8, out=out2)
        live = ~torch.isnan(out[1])
        assert torch.equal(out2[1][live].view(torch.int16), out[1][live].view(torch.int16)), "contamination"
    print("scores OK")

# ------------------------------------------------------------------ blocks
if section("blocks"):
    for trial in range(6):
        rows, width, blocks, block = 3, 512, random.choice((4, 8, 64)), 8
        scores = (torch.randn(rows, width) * 2).half()
        if trial % 2:
            scores = scores.clamp(-1, 1).round()  # many ties
        n = torch.tensor([random.randint(1, width) for _ in range(rows)], dtype=torch.int64)
        n[0] = width if trial == 0 else n[0]
        garbage = scores.clone()
        for t in range(rows):
            garbage[t, int(n[t]):] = float("nan")  # unset columns past the length
        got = rx.decode_candidates(garbage, n, blocks=blocks, block=block, topk_fn=lambda *a: topk_mod.index_topk(*a))
        for t in range(rows):
            m = int(n[t])
            ref = upstream_candidate_blocks(scores[t, :m].float(), torch.tensor(m), blocks, block)
            ref_blocks = sorted({p // block for p in range(m) if ref[p]})
            mine = [int(b) for b in got[t] if b >= 0]
            assert mine == ref_blocks, ("blocks", trial, t, mine[:8], ref_blocks[:8])
            assert (got[t, len(mine):] == -1).all()
    print("blocks OK")

# ------------------------------------------------------------------ prefill
# The overlay harness loaded upstream attention_ops with VLLM_V41_INDEX_TOPK="0" (its envs stub) and passed
# fused_topk=False to match; the vendored attention_ops carries the production value (INDEX_TOPK=1), so the
# selector under test uses the same flag as its reference.
if section("prefill"):
    for trial, (start, count, blocks) in enumerate(((0, 40, 2048), (1300, 25, 128), (2100, 30, 128))):
        total = start + count
        keys16 = (torch.randn(total, 128) * 0.5).half()
        positions = torch.arange(start, total, dtype=torch.int64)
        valid = torch.ones(count, dtype=torch.bool)
        q, w = queries(count)
        keys = torch.zeros(max(total, 512), 128).half()
        keys[:total] = keys16
        ref_full = ops.select_compressed_positions(q, w, keys, positions, valid, 1, lambda x: x)
        got_full = rx.prefill_select(q, w, keys, positions, valid, 1, lambda x: x, kind="full",
                                     blocks=blocks, block=8, fused_topk=ops._FUSED_TOPK)
        assert torch.equal(ref_full, got_full), "prefill full"
        cand = torch.full((count, blocks), -2, dtype=torch.int32)
        got_src = rx.prefill_select(q, w, keys, positions, valid, 1, lambda x: x, kind="source",
                                    blocks=blocks, block=8, cand_out=cand, fused_topk=ops._FUSED_TOPK)
        assert torch.equal(ref_full, got_src), "prefill source top-k"
        q2, w2 = queries(count)
        got_re = rx.prefill_select(q2, w2, keys, positions, valid, 1, lambda x: x, kind="reindex",
                                   blocks=blocks, block=8, cand_in=cand, fused_topk=ops._FUSED_TOPK)
        # reference: upstream Indexer.forward semantics on the same FP16 scores
        def tile_scores(qq, ww):
            s = torch.mm(qq.reshape(-1, 128), keys.T).view(count, H, -1)
            return s.relu_().mul_(ww[:, :, None]).sum(1)

        all1, all2 = tile_scores(q, w), None
        for t in range(count):
            m = int(positions[t]) + 1
            s1 = all1[t, :m]
            keep = upstream_candidate_blocks(s1.float(), torch.tensor(m), blocks, 8)
            mine = sorted({int(c) for c in cand[t] if c >= 0})
            assert mine == sorted({p // 8 for p in range(m) if keep[p]}), ("prefill cand", trial, t)
            if all2 is None:
                all2 = tile_scores(q2, w2)
            s2 = all2[t, :m].clone()
            s2 = s2.masked_fill(~keep, float("-inf"))
            k = min(512, m)
            want = set(s2.topk(k).indices.tolist()) if blocks < 2048 else None
            got = [int(i) for i in got_re[t] if i >= 0]
            # every pick inside the candidates; when fewer than 512 candidates are
            # reachable, all of them are picked
            assert all(keep[i] for i in got), ("prefill reindex outside pool", trial, t)
            if int(keep.sum()) <= 512:
                assert sorted(got) == sorted(int(i) for i in torch.nonzero(keep).flatten()), ("pool", trial, t)
            else:
                assert len(got) == k
                # same selected score multiset (FP16 ties may swap equal rows)
                a = sorted(s2[got].tolist())
                b = sorted(s2[sorted(want)].tolist())
                assert a == b, ("prefill reindex", trial, t)
        if blocks >= 2048:
            ref_re = ops.select_compressed_positions(q2, w2, keys, positions, valid, 1, lambda x: x)
            assert torch.equal(ref_re, got_re), "prefill reindex with all blocks == full"
    print("prefill OK")


# ------------------------------------------------------------------ select glue
class _Lin:
    def __init__(self, a, b):
        self.w = (torch.randn(a, b) / a**0.5).float()

    def __call__(self, x):
        return (x.float() @ self.w).half()


def _layer(layer):
    indexer = types.SimpleNamespace(wq_b=_Lin(64, H * 128), weights_proj=_Lin(32, H), heads=H)
    return types.SimpleNamespace(layer=layer, ratio=1, source=20, indexer=indexer)


def _query(lay, x, qrv, T):
    """The overlay's in-selector indexer query (RoPE stubbed to identity there) and weights."""
    q = qdq(qr, lay.indexer.wq_b(qrv).view(T, H, 128).contiguous(), fp4=True, group=32)
    return q, lay.indexer.weights_proj(x) * (1.0 / 64.0)


def _ident(x):
    return x


if section("select"):
    for blocks in (2048, 128):
        cfg = types.SimpleNamespace(candidate_source_layer_id=20, candidate_topk_blocks=blocks,
                                    candidate_block_size=8)
        cand_source, blocks_cfg, block_cfg = rx.hierarchy(cfg)
        assert (cand_source, blocks_cfg, block_cfg) == (20, blocks, 8)
        # decode tokens at 3 contexts + one prefill request of 20 tokens + one decode
        lengths = [1, 20, 1, 1]
        starts = [2299, 1450, 39, 1033]
        totals = [s + n for s, n in zip(starts, lengths)]
        cache, table, _ = build_pages(LAYOUT, 20, totals)
        qsl, positions, valid, token_req = make_batch(lengths, starts)
        T = positions.numel()
        prefill = [(1, int(qsl[1]), int(qsl[2]))]
        decode_index = torch.tensor([t for t in range(T) if not qsl[1] <= t < qsl[2]], dtype=torch.long)
        step = types.SimpleNamespace(prefill=prefill, decode_index=decode_index, valid=valid,
                                     token_req=token_req, qsl=qsl, qsl_cpu=qsl, num_reqs=4)
        x = torch.randn(T, 32).half()
        qrv = torch.randn(T, 64).half()
        src, user = _layer(20), _layer(24)
        assert rx.role(20, cand_source) == "source" and rx.role(24, cand_source) == "reindex"
        common = {"layout": LAYOUT, "source": 20, "ratio": 1, "width": 2560}
        q_src, w_src = _query(src, x, qrv, T)
        q_user, w_user = _query(user, x, qrv, T)
        got_src, cand = rx.select(rx.role(20, cand_source), q_src, w_src, positions, step, cache, table,
                                  blocks=blocks, block=8, all_reduce=_ident, **common)
        ref_src = pk.select_positions(q_src, w_src, positions, step, cache, table, all_reduce=_ident, **common)
        assert torch.equal(got_src, ref_src), ("select source", blocks)
        got_user, cand_back = rx.select(rx.role(24, cand_source), q_user, w_user, positions, step, cache,
                                        table, blocks=blocks, block=8, all_reduce=_ident, candidates=cand,
                                        **common)
        assert cand_back is cand
        ref87 = pk.select_positions(q_user, w_user, positions, step, cache, table, all_reduce=_ident, **common)
        if blocks == 2048:
            assert torch.equal(got_user, ref87), "select reindex (all blocks) == 87"
        else:
            # upstream reference per decode token: 87's full scores of both layers
            for t in decode_index.tolist():
                n = int(positions[t]) + 1

                def full_scores(q, wt):
                    out = torch.full((T, 2560), float("nan")).half()
                    pk.index_scores(q, wt, cache, table, positions, valid, token_req,
                                    layout=LAYOUT, layer=20, out=out)
                    return out[t, :n]

                keep = upstream_candidate_blocks(full_scores(q_src, w_src).float(), torch.tensor(n), blocks, 8)
                assert sorted(int(c) for c in cand[t] if c >= 0) == sorted({p // 8 for p in range(n) if keep[p]})
                s2 = full_scores(q_user, w_user).masked_fill(~keep, float("-inf"))
                ref = torch.full((1, 512), -1, dtype=torch.int32)
                topk_mod.index_topk(ref, s2[None].contiguous(), positions[t : t + 1], valid[t : t + 1], 1, 512)
                assert torch.equal(got_user[t], ref[0]), ("select reindex", t)
            for t in range(int(qsl[1]), int(qsl[2])):
                got = [int(i) for i in got_user[t] if i >= 0]
                pool = {int(c) for c in cand[t] if c >= 0}
                assert got and all(i // 8 in pool for i in got), ("prefill pool", t)
            assert not torch.equal(got_user, ref87), "small pool must change the selection"
        # a reindex layer without its source in this forward raises
        try:
            rx.select("reindex", q_user, w_user, positions, step, cache, table, blocks=blocks, block=8,
                      all_reduce=_ident, candidates=None, **common)
        except RuntimeError as e:
            assert "no candidates" in str(e)
        else:
            raise AssertionError("missing candidates not detected")
    print("select OK")

print("ALL OK")

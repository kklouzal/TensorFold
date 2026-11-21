"""MU state / paged kernels vs the vendored upstream single-sequence kernels (ported from the friend overlay
snapshot lp-snap-c1es8-numa-admit2-imgtext-share2-fix5-ram4-stepbud-ram3-memfix-ctx1m-ppovl-opt2-pfx-1-eng-pfxfix4, tests/ds41mu_interpret_check.py;
imports adapted, sections, oracles and tolerances verbatim).

    CPU: TRITON_INTERPRET=1 CUDA_VISIBLE_DEVICES= python mu_interpret_check.py [section ...]
    GPU: DS41MU_DEVICE=cuda python mu_interpret_check.py [section ...]
Prints "<section> OK" per section and "ALL OK".
"""
# ruff: noqa: B023  -- the overlay's per-iteration helpers are called within the iteration they close over.

import random
import sys

import torch
import torch.nn.functional as F  # noqa: F401
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

torch.manual_seed(0)
random.seed(0)
SECTIONS = set(sys.argv[1:])


def triton_np2(n):
    import triton

    return triton.next_power_of_2(n)


def section(name):
    return not SECTIONS or name in SECTIONS


def make_batch(lengths, starts, slots, pad_tokens=0):
    """Flattened batch: request r has lengths[r] tokens from position starts[r]."""
    qsl = [0]
    positions = []
    for n, s in zip(lengths, starts):
        positions += list(range(s, s + n))
        qsl.append(qsl[-1] + n)
    total = qsl[-1]
    positions += [0] * pad_tokens
    valid = torch.zeros(total + pad_tokens, dtype=torch.bool)
    valid[:total] = True
    return (
        torch.tensor(qsl, dtype=torch.int32),
        torch.tensor(positions, dtype=torch.int64),
        valid,
        torch.tensor(slots, dtype=torch.int32),
    )


# ------------------------------------------------------------ token_req
if section("token_req"):
    st = load_payload("local_mu_state")
    for _ in range(30):
        r = random.randint(1, 9)
        lengths = [random.choice([0, 1, 1, 2, 5, 17]) for _ in range(r)]
        qsl = torch.tensor([0] + list(torch.tensor(lengths).cumsum(0)), dtype=torch.int32)
        total = int(qsl[-1])
        pad = random.randint(0, 5)
        out = torch.full((total + pad + 3,), 77, dtype=torch.int32)
        got = st.build_token_req(qsl, r, total + pad, out)
        ref = st.token_req_reference(qsl, r, total + pad)
        assert torch.equal(got, ref), (lengths, got, ref)
        assert int(out[total + pad]) == 77  # nothing written past num_tokens
    print("token_req OK")


# ------------------------------------------------------------ Engram hashes
if section("engram"):
    st = load_payload("local_mu_state")
    up = load_upstream("vllm/model_executor/layers/deepseek_v41_engram.py")
    VOCAB = 300
    token_map = torch.randint(0, 120, (VOCAB,), dtype=torch.int32)
    primes = torch.tensor(
        [[random.choice([10007, 10009, 10037, 10039]) for _ in range(24)] for _ in range(2)],
        dtype=torch.int64,
    )
    offsets = torch.randint(0, 1 << 20, (2, 24), dtype=torch.int64)
    multipliers = torch.randint(1, 1 << 30, (2, 4), dtype=torch.int64)
    pad_id = 2
    pad = token_map[pad_id : pad_id + 1].long()
    S = 6

    def upstream_request(ids, positions, valid, history_row):
        """Upstream kernels on one request alone (its own history)."""
        n = ids.numel()
        history = history_row.clone()
        metadata = torch.zeros(3, dtype=torch.int64)
        hashes = torch.empty(n, 2, 24, dtype=torch.int64)
        candidates = torch.empty(n + 1, 3, dtype=torch.int64)
        meta = {"N": n, "VOCAB": VOCAB, "MASKED": True}
        up._hash_request[(1,)](
            ids, positions, token_map, valid, history, metadata,
            MAX_CONTEXT=1 << 20, BLOCK=triton_next(n), **meta,
        )
        up._hash_rows[(n, 2)](
            ids, positions, valid, token_map, primes, offsets, multipliers, pad,
            history, hashes, **meta,
        )
        up._history_candidates[(n + 1,)](
            ids, positions, valid, token_map, history, candidates, **meta
        )
        up._commit_history[(1,)](
            candidates, metadata, history, torch.tensor([n], dtype=torch.int64)
        )
        return hashes, history

    import triton

    def triton_next(n):
        return triton.next_power_of_2(n)

    for trial in range(12):
        r = random.randint(1, 5)
        lengths = [random.choice([1, 1, 2, 3, 4, 7, 12]) for _ in range(r)]
        slots = random.sample(range(S), r)
        state = st.MuEngramHashState(
            token_map, primes, offsets, multipliers, slots=S, pad_token_id=pad_id,
            max_tokens=256, max_context=1 << 20, trap=False,
        )
        starts = []
        for rr in range(r):
            if random.random() < 0.4:
                starts.append(0)
            else:
                s = random.randint(1, 50)
                starts.append(s)
                hist = torch.randint(-1, 120, (3,), dtype=torch.int64)
                state.history[slots[rr]] = torch.cat([hist, torch.tensor([s])])
        pad_tokens = random.randint(0, 3)
        qsl, positions, valid, req_slot = make_batch(lengths, starts, slots, pad_tokens)
        ids = torch.randint(0, VOCAB, (positions.numel(),), dtype=torch.int64)
        token_req = st.build_token_req(qsl, r, positions.numel(), torch.empty(positions.numel(), dtype=torch.int32))
        before = state.history.clone()
        hashes = state(ids, positions, valid, token_req, qsl, req_slot)
        assert int(state.err) == 0, int(state.err)
        ref = st.hash_reference(
            ids, positions, valid, qsl, r, req_slot, before, token_map, primes,
            offsets, multipliers, pad,
        )
        total = int(qsl[-1])
        assert torch.equal(hashes[:total], ref[:total]), "hash vs reference"
        state.commit_all(ids, positions, valid, qsl, req_slot, r, token_req)
        ref_hist = st.commit_history_reference(
            ids, positions, valid, qsl, r, req_slot, before, token_map
        )
        assert torch.equal(state.history, ref_hist), "commit vs reference"
        for rr in range(r):
            s0, s1 = int(qsl[rr]), int(qsl[rr + 1])
            u_hash, u_hist = upstream_request(
                ids[s0:s1].contiguous(), positions[s0:s1].contiguous(),
                valid[s0:s1].contiguous(), before[slots[rr]],
            )
            assert torch.equal(hashes[s0:s1], u_hash), ("upstream hash", trial, rr)
            assert torch.equal(state.history[slots[rr]], u_hist), ("upstream commit", trial, rr)
        # untouched slots unchanged
        for s in set(range(S + 1)) - set(slots):
            assert torch.equal(state.history[s], before[s])
        # isolation: other requests' ids and other slots' history replaced
        target = random.randrange(r)
        ids2 = torch.randint(0, VOCAB, ids.shape, dtype=torch.int64)
        s0, s1 = int(qsl[target]), int(qsl[target + 1])
        ids2[s0:s1] = ids[s0:s1]
        state2 = st.MuEngramHashState(
            token_map, primes, offsets, multipliers, slots=S, pad_token_id=pad_id,
            max_tokens=256, max_context=1 << 20, trap=False,
        )
        state2.history.copy_(torch.randint(-1, 120, before.shape))
        state2.history[slots[target]] = before[slots[target]]
        for rr in range(r):  # keep other continuing requests valid
            if starts[rr]:
                state2.history[slots[rr], 3] = starts[rr]
        h2 = state2(ids2, positions, valid, token_req, qsl, req_slot)
        assert torch.equal(h2[s0:s1], hashes[s0:s1]), "cross-slot contamination"
    # validation: stale continuing start, non-contiguous positions, bad id
    state = st.MuEngramHashState(
        token_map, primes, offsets, multipliers, slots=S, pad_token_id=pad_id,
        max_tokens=64, max_context=1 << 20, trap=False,
    )
    qsl, positions, valid, req_slot = make_batch([2, 3], [0, 9], [0, 1])
    token_req = st.build_token_req(qsl, 2, 5, torch.empty(5, dtype=torch.int32))
    ids = torch.randint(0, VOCAB, (5,))
    state(ids, positions, valid, token_req, qsl, req_slot)
    assert int(state.err) & st.ERR_CONTINUE
    state.err.zero_()
    state.history[1, 3] = 9
    positions[3] = 40
    state(ids, positions, valid, token_req, qsl, req_slot)
    assert int(state.err) == st.ERR_POSITION
    state.err.zero_()
    positions[3] = 10
    ids[0] = VOCAB + 5
    state(ids, positions, valid, token_req, qsl, req_slot)
    assert int(state.err) == st.ERR_TOKEN
    print("engram OK")


# ------------------------------------------------------------ C2 pooling
if section("c2"):
    st = load_payload("local_mu_state")
    up = load_upstream("vllm/model_executor/kernels/deepseek_v41_compression.py")
    S = 5
    norm = torch.rand(512) + 0.5

    def upstream_request(kv, score, positions, partial_row, committed):
        n = kv.shape[0]
        partial = partial_row.clone()
        candidates = torch.zeros(7, 2, 512)
        metadata = torch.zeros(3, dtype=torch.int64)
        position = torch.tensor([committed], dtype=torch.int64)
        out = torch.empty(n, 512, dtype=torch.float16)
        complete = torch.empty(n, dtype=torch.bool)
        import triton

        up._begin[(1,)](positions, position, metadata, N=n, BLOCK=triton.next_power_of_2(n))
        up._pool_c2[(n,)](
            kv, score, positions, partial, candidates, norm, out, complete,
            N=n, D=512, EPS=1e-20, BLOCK=512,
            enable_fp_fusion=False,  # as upstream's own launch (compression.py C2 pool): no FMA contraction
        )
        up._commit_c2[(1,)](
            partial, candidates, metadata, position, torch.tensor([n]), D=512, BLOCK=512
        )
        return out, complete, partial, int(position)

    for trial in range(10):
        r = random.randint(1, 4)
        lengths = [random.choice([1, 1, 2, 3, 6, 9]) for _ in range(r)]
        slots = random.sample(range(S), r)
        state = st.MuC2State(norm, slots=S, trap=False)
        starts = []
        for rr in range(r):
            s = 0 if random.random() < 0.3 else random.randint(1, 40)
            starts.append(s)
            state.committed[slots[rr]] = s
            if s % 2 == 1:  # an open group: its first token is the partial row
                state.partial[slots[rr]] = torch.randn(2, 512)
        qsl, positions, valid, req_slot = make_batch(lengths, starts, slots, random.randint(0, 2))
        n = positions.numel()
        kv, score = torch.randn(n, 512), torch.randn(n, 512)
        token_req = st.build_token_req(qsl, r, n, torch.empty(n, dtype=torch.int32))
        before = state.partial.clone()
        out, complete = state(kv, score, positions, token_req, qsl, req_slot)
        assert int(state.err) == 0
        state.commit_all(kv, score, positions, qsl, req_slot, r, token_req)
        ref = st.pool_c2_reference(kv, score, positions, qsl, r, req_slot, before, norm)
        total = int(qsl[-1])
        assert (out[:total].float() - ref[:total].float()).abs().max() < 2e-2
        for rr in range(r):
            s0, s1 = int(qsl[rr]), int(qsl[rr + 1])
            u_out, u_done, u_partial, u_pos = upstream_request(
                kv[s0:s1].contiguous(), score[s0:s1].contiguous(),
                positions[s0:s1].contiguous(), before[slots[rr]], starts[rr],
            )
            done = complete[s0:s1]
            assert torch.equal(done, u_done)
            # upstream leaves incomplete rows as the normalised zero vector
            assert torch.equal(out[s0:s1], u_out), ("upstream pool", trial, rr)
            assert torch.equal(state.partial[slots[rr]], u_partial), ("upstream commit", trial, rr)
            assert int(state.committed[slots[rr]]) == u_pos
        for s in set(range(S + 1)) - set(slots):
            assert torch.equal(state.partial[s], before[s])
        # isolation
        target = random.randrange(r)
        s0, s1 = int(qsl[target]), int(qsl[target + 1])
        kv2, score2 = torch.randn(n, 512), torch.randn(n, 512)
        kv2[s0:s1], score2[s0:s1] = kv[s0:s1], score[s0:s1]
        state2 = st.MuC2State(norm, slots=S, trap=False)
        state2.partial.copy_(torch.randn_like(state2.partial))
        state2.partial[slots[target]] = before[slots[target]]
        state2.committed.copy_(state.committed * 0)
        for rr in range(r):
            state2.committed[slots[rr]] = starts[rr]
        out2, _ = state2(kv2, score2, positions, token_req, qsl, req_slot)
        assert torch.equal(out2[s0:s1], out[s0:s1]), "cross-slot contamination"
    # stale partial: continuing odd start with a different committed position
    state = st.MuC2State(norm, slots=S, trap=False)
    qsl, positions, valid, req_slot = make_batch([1], [7], [2])
    state.committed[2] = 5
    token_req = st.build_token_req(qsl, 1, 1, torch.empty(1, dtype=torch.int32))
    state(torch.randn(1, 512), torch.randn(1, 512), positions, token_req, qsl, req_slot)
    assert int(state.err) & st.ERR_CONTINUE
    print("c2 OK")

# ------------------------------------------------------------ paged FP16 ops
def paged_layout(lengths_total, block_size, pages_total, max_blocks):
    """Random disjoint physical pages per request; -1 past each request's need."""
    pool = list(range(1, pages_total))
    random.shuffle(pool)
    table = torch.full((len(lengths_total), max_blocks), -1, dtype=torch.int32)
    for r, total in enumerate(lengths_total):
        need = (total + block_size - 1) // block_size
        for b in range(need):
            table[r, b] = pool.pop()
    return table


if section("paged"):
    st = load_payload("local_mu_state")
    pg = load_payload("local_mu_paged")
    up = load_attention_ops()
    BS = 16
    for trial in range(6):
        r = random.randint(1, 4)
        starts = [random.randint(0, 300) for _ in range(r)]
        lengths = [random.choice([1, 1, 2, 5, 9]) for _ in range(r)]
        slots = list(range(r))
        qsl, positions, valid, _ = make_batch(lengths, starts, slots, random.randint(0, 2))
        n = positions.numel()
        token_req = st.build_token_req(qsl, r, n, torch.empty(n, dtype=torch.int32))
        totals = [s + l for s, l in zip(starts, lengths)]
        table = paged_layout(totals, BS, 200, 32)
        for ratio, dim in ((1, 512), (2, 512), (2, 128), (1, 128)):
            cache = torch.randn(200, BS // ratio, dim).half()
            values = torch.randn(n, dim).half()
            mine = cache.clone()
            pg.mu_insert_latents(mine, values, positions, valid, token_req, table, block_size=BS, ratio=ratio)
            ref = cache.clone()
            for rr in range(r):
                s0, s1 = int(qsl[rr]), int(qsl[rr + 1])
                up._insert[(s1 - s0,)](
                    values[s0:s1].contiguous(), positions[s0:s1].contiguous(),
                    valid[s0:s1].contiguous(), table[rr : rr + 1].contiguous(), ref,
                    STRIDE=ref.stride(0), BLOCK_SIZE=BS, RATIO=ratio, D=dim,
                    BLOCK=triton_np2(dim),
                )
            assert torch.equal(mine, ref), ("insert", trial, ratio, dim)
        # gather window + compressed (ratio 2) for every token
        window_cache = torch.randn(200, BS, 512).half()
        comp_cache = torch.randn(200, BS // 2, 512).half()
        comp_table = paged_layout(totals, BS, 200, 32)
        K = 8
        topk = torch.full((n, K), -1, dtype=torch.int32)
        for t in range(n):
            live = (int(positions[t]) + 1) // 2
            pick = random.sample(range(max(live, 1)), min(K, live)) if live else []
            topk[t, : len(pick)] = torch.tensor(sorted(pick), dtype=torch.int32)
        keys = torch.empty(n, 128 + K, 512).half()
        unused = torch.empty(n, 128 + K, dtype=torch.bool)
        pg.mu_gather_attention(
            keys, unused, positions, valid, token_req, window_cache, table,
            window_block=BS, compressed_cache=comp_cache, compressed_table=comp_table,
            compressed_block=BS, ratio=2, topk=topk,
        )
        for rr in range(r):
            s0, s1 = int(qsl[rr]), int(qsl[rr + 1])
            k2 = torch.empty(s1 - s0, 128 + K, 512).half()
            u2 = torch.empty(s1 - s0, 128 + K, dtype=torch.bool)
            up._gather_attention[(s1 - s0, 128 + K)](
                window_cache, table[rr : rr + 1].contiguous(), comp_cache,
                comp_table[rr : rr + 1].contiguous(), positions[s0:s1].contiguous(),
                valid[s0:s1].contiguous(), topk[s0:s1].contiguous(), k2, u2,
                WINDOW_STRIDE=window_cache.stride(0), COMPRESSED_STRIDE=comp_cache.stride(0),
                WINDOW_BLOCK=BS, COMPRESSED_BLOCK=BS, RATIO=2, WINDOW=128, TOPK=K,
            )
            assert torch.equal(unused[s0:s1], u2), ("gather mask", trial, rr)
            assert torch.equal(keys[s0:s1], k2), ("gather keys", trial, rr)
        total = int(qsl[-1])
        assert bool(unused[total:].all()), "padding tokens must gather nothing"
        # decode indexer scores over paged FP16 index keys
        H = 8
        for ratio in (2, 1):
            icache = (torch.randn(200, BS // ratio, 128) * 0.3).half()
            itable = paged_layout(totals, BS, 200, 32)
            q = (torch.randn(n, H, 128) * 0.3).half()
            w = torch.randn(n, H).half()
            width = 400 // ratio
            out = torch.full((n, width), float("nan")).half()
            pg.mu_index_scores(q, w, icache, itable, positions, valid, token_req, block_size=BS, ratio=ratio, out=out)
            ref = pg.index_scores_reference(q, w, icache, itable, positions, valid, token_req, block_size=BS, ratio=ratio, width=width)
            for t in range(total):
                end = (int(positions[t]) + 1) // ratio
                a, b = out[t, :end].float(), ref[t, :end].float()
                assert torch.allclose(a, b, rtol=4e-3, atol=4e-3), ("scores", trial, t, (a - b).abs().max())
            # upstream path for one request (gather keys + mm), loose: fp16 sums
            rr = random.randrange(r)
            s0, s1 = int(qsl[rr]), int(qsl[rr + 1])
            keys16 = torch.zeros(width, 128).half()
            end = (int(positions[s1 - 1]) + 1) // ratio
            up._gather_index[(width,)](
                icache, itable[rr : rr + 1].contiguous(), positions[s0:s1].contiguous(),
                valid[s0:s1].contiguous(), keys16, STRIDE=icache.stride(0),
                BLOCK_SIZE=BS, RATIO=ratio, TOKENS=s1 - s0,
            )
            ups = (q[s0:s1].float().reshape(-1, 128) @ keys16.float().T).half().float()
            ups = (ups.relu().view(s1 - s0, H, -1) * w[s0:s1].float()[:, :, None]).half().float().sum(1)
            for t in range(s0, s1):
                e = (int(positions[t]) + 1) // ratio
                assert torch.allclose(out[t, :e].float(), ups[t - s0, :e], rtol=1e-2, atol=1e-2)
            # isolation: other requests' key pages replaced, same tables
            icache2 = (torch.randn_like(icache.float()) * 0.3).half()
            used = itable[rr][itable[rr] >= 0].long()
            icache2[used] = icache[used]
            q2, w2 = q.clone(), w.clone()
            other = torch.ones(n, dtype=torch.bool)
            other[s0:s1] = False
            q2[other] = (torch.randn(int(other.sum()), H, 128) * 0.3).half()
            out2 = torch.full((n, width), float("nan")).half()
            pg.mu_index_scores(q2, w2, icache2, itable, positions, valid, token_req, block_size=BS, ratio=ratio, out=out2)
            for t in range(s0, s1):
                e = (int(positions[t]) + 1) // ratio
                assert torch.equal(out2[t, :e], out[t, :e]), "cross-request contamination"
    print("paged OK")

# ------------------------------------------------------------ hardening
if section("hardening"):
    # CUDA-graph padding and dummy batches must never touch a real slot or the
    # scratch slot, and must not raise validation bits.
    st = load_payload("local_mu_state")
    VOCAB = 300
    token_map = torch.randint(0, 120, (VOCAB,), dtype=torch.int32)
    primes = torch.full((2, 24), 10007, dtype=torch.int64)
    offsets = torch.zeros(2, 24, dtype=torch.int64)
    multipliers = torch.randint(1, 1 << 30, (2, 4), dtype=torch.int64)
    S = 4
    norm = torch.rand(512) + 0.5
    for case in ("padded", "dummy"):
        hs = st.MuEngramHashState(
            token_map, primes, offsets, multipliers, slots=S, max_tokens=64,
            max_context=4096, trap=False,
        )
        c2 = st.MuC2State(norm, slots=S, trap=False)
        hs.history.copy_(torch.randint(-1, 120, hs.history.shape))
        hs.history[:, 3] = torch.tensor([7, 3, 0, 9, 0])
        c2.partial.copy_(torch.randn_like(c2.partial))
        c2.committed.copy_(torch.tensor([7, 3, 0, 9, 0]))
        hist0, part0, comm0 = hs.history.clone(), c2.partial.clone(), c2.committed.clone()
        if case == "padded":
            # 2 real decodes (slots 0, 1) + 2 zero-length padded request rows
            # + 2 padding tokens with garbage positions (FULL graph replay).
            qsl = torch.tensor([0, 1, 2, 2, 2], dtype=torch.int32)
            positions = torch.tensor([7, 3, 10**6, 5], dtype=torch.int64)
            valid = torch.tensor([True, True, False, False])
            req_slot = torch.tensor([0, 1, S, S, S, S], dtype=torch.int32)
            token_req = st.build_token_req(qsl, 4, 4, torch.empty(4, dtype=torch.int32))
            assert token_req.tolist() == [0, 1, -1, -1]
            num_reqs = 4
        else:
            # capture / DP dummy: token_req all -1, req_slot still holds the
            # previous step's real slots, query starts describe 3 requests.
            qsl = torch.tensor([0, 1, 2, 3], dtype=torch.int32)
            positions = torch.tensor([0, 10**6, 5], dtype=torch.int64)
            valid = torch.zeros(3, dtype=torch.bool)
            req_slot = torch.tensor([0, 1, 3, S], dtype=torch.int32)
            token_req = torch.full((3,), -1, dtype=torch.int32)
            num_reqs = 3
        n = positions.numel()
        ids = torch.randint(0, VOCAB, (n,), dtype=torch.int64)
        hs(ids, positions, valid, token_req, qsl, req_slot)
        hs.commit_all(ids, positions, valid, qsl, req_slot, num_reqs, token_req)
        kv, score = torch.randn(n, 512), torch.randn(n, 512)
        c2(kv, score, positions, token_req, qsl, req_slot)
        c2.commit_all(kv, score, positions, qsl, req_slot, num_reqs, token_req)
        assert int(hs.err) == 0 and int(c2.err) == 0, (case, int(hs.err), int(c2.err))
        assert torch.equal(hs.history[S], hist0[S]) and torch.equal(c2.partial[S], part0[S]), case
        untouched = [2, 3] if case == "padded" else [0, 1, 2, 3]
        for s in untouched:
            assert torch.equal(hs.history[s], hist0[s]), (case, "history", s)
            assert torch.equal(c2.partial[s], part0[s]), (case, "partial", s)
            assert int(c2.committed[s]) == int(comm0[s]), (case, "committed", s)
        if case == "padded":
            assert int(hs.history[0, 3]) == 8 and int(hs.history[1, 3]) == 4
    print("hardening OK")

# ------------------------------------------------------------ prefix restore
if section("prefix"):
    # Patch 90: a fresh slot restored at a cache hit h continues exactly like
    # the slot that computed the h cached tokens (Engram hashes + history, C2
    # pooled values + partial row + committed position).
    import types as _types

    st = load_payload("local_mu_state")
    mm = load_payload("local_mu_model")
    VOCAB = 300
    token_map = torch.randint(0, 120, (VOCAB,), dtype=torch.int32)
    primes = torch.tensor([[10007] * 24, [10009] * 24], dtype=torch.int64)
    offsets = torch.randint(0, 1 << 20, (2, 24), dtype=torch.int64)
    multipliers = torch.randint(1, 1 << 30, (2, 4), dtype=torch.int64)
    norm = torch.rand(512) + 0.5
    S = 4
    for h, chunks in ((2, [2, 5]), (8, [3, 5, 4]), (16, [16, 1, 1])):
        total = sum(chunks)
        ids = torch.randint(0, VOCAB, (total,), dtype=torch.int64)
        kv, score = torch.randn(total, 512), torch.randn(total, 512)

        def run(slot, start, pieces, hs, c2):
            outs_h, outs_c = {}, {}
            pos = start
            for n in pieces:
                qsl = torch.tensor([0, n], dtype=torch.int32)
                positions = torch.arange(pos, pos + n, dtype=torch.int64)
                valid = torch.ones(n, dtype=torch.bool)
                req_slot = torch.tensor([slot], dtype=torch.int32)
                token_req = st.build_token_req(qsl, 1, n, torch.empty(n, dtype=torch.int32))
                hashes = hs(ids[pos : pos + n], positions, valid, token_req, qsl, req_slot)
                hs.commit_all(ids[pos : pos + n], positions, valid, qsl, req_slot, 1, token_req)
                out, _ = c2(kv[pos : pos + n], score[pos : pos + n], positions, token_req, qsl, req_slot)
                c2.commit_all(kv[pos : pos + n], score[pos : pos + n], positions, qsl, req_slot, 1, token_req)
                for i in range(n):
                    outs_h[pos + i], outs_c[pos + i] = hashes[i], out[i]
                pos += n
            assert int(hs.err) == 0 and int(c2.err) == 0
            return outs_h, outs_c

        def fresh():
            return (
                st.MuEngramHashState(token_map, primes, offsets, multipliers, slots=S,
                                     max_tokens=64, max_context=4096, trap=False),
                st.MuC2State(norm, slots=S, trap=False),
            )

        hs_a, c2_a = fresh()
        full_h, full_c = run(1, 0, chunks, hs_a, c2_a)
        # B: the first h tokens came from the prefix cache; slot 2 restored
        hs_b, c2_b = fresh()
        ctx = _types.SimpleNamespace(hash_state=hs_b, layers_with_c2=[c2_b])
        mm.restore_slot(ctx.hash_state, ctx.layers_with_c2, 2, h, ids.tolist())
        rest = []
        left = total - h
        for n in chunks:
            if left <= 0:
                break
            rest.append(min(n, left))
            left -= rest[-1]
        part_h, part_c = run(2, h, rest, hs_b, c2_b)
        for p in range(h, total):
            assert torch.equal(part_h[p], full_h[p]), ("restored hash", h, p)
            assert torch.equal(part_c[p], full_c[p]), ("restored c2", h, p)
        assert torch.equal(hs_b.history[2], hs_a.history[1]), ("history", h)
        assert torch.equal(c2_b.partial[2], c2_a.partial[1]) and int(c2_b.committed[2]) == int(c2_a.committed[1])
    print("prefix OK")

# ------------------------------------------------------------ speculation state
if section("spec"):
    # Phase 4 groundwork: per-slot candidate snapshots + commit by accepted
    # count == upstream's single-sequence candidates/commit, per request.
    import triton

    st = load_payload("local_mu_state")
    up_e = load_upstream("vllm/model_executor/layers/deepseek_v41_engram.py")
    up_c = load_upstream("vllm/model_executor/kernels/deepseek_v41_compression.py")
    VOCAB = 300
    token_map = torch.randint(0, 120, (VOCAB,), dtype=torch.int32)
    primes = torch.full((2, 24), 10007, dtype=torch.int64)
    offsets = torch.zeros(2, 24, dtype=torch.int64)
    multipliers = torch.randint(1, 1 << 30, (2, 4), dtype=torch.int64)
    norm = torch.rand(512) + 0.5
    S, K = 5, 6
    for trial in range(8):
        r = random.randint(1, 4)
        lengths = [random.choice([1, 2, 4, 6, 9]) for _ in range(r)]
        slots = random.sample(range(S), r)
        starts = [random.choice([0, random.randint(1, 40)]) for _ in range(r)]
        hs = st.MuEngramHashState(token_map, primes, offsets, multipliers, slots=S,
                                  max_tokens=64, max_context=4096, trap=False)
        c2 = st.MuC2State(norm, slots=S, trap=False)
        for rr in range(r):
            if starts[rr]:
                hs.history[slots[rr]] = torch.cat([torch.randint(-1, 120, (3,)), torch.tensor([starts[rr]])])
                c2.committed[slots[rr]] = starts[rr]
                if starts[rr] % 2:
                    c2.partial[slots[rr]] = torch.randn(2, 512)
        qsl, positions, valid, req_slot = make_batch(lengths, starts, slots, 1)
        n = positions.numel()
        token_req = st.build_token_req(qsl, r, n, torch.empty(n, dtype=torch.int32))
        ids = torch.randint(0, VOCAB, (n,), dtype=torch.int64)
        kv, score = torch.randn(n, 512), torch.randn(n, 512)
        before_h, before_p = hs.history.clone(), c2.partial.clone()
        spec = st.MuSpeculation(hs, [c2], slots=S, max_inputs=K)
        spec.snapshot((ids, positions, valid, [(kv, score)]), qsl, req_slot, token_req, r)
        counts = torch.tensor(
            [random.randint(0, ln) if ln <= K else ln for ln in lengths], dtype=torch.int32
        )
        spec.commit(req_slot, counts, r)
        assert int(hs.err) == 0 and int(c2.err) == 0
        for rr in range(r):
            s0, s1 = int(qsl[rr]), int(qsl[rr + 1])
            m = s1 - s0
            # upstream Engram, this request alone
            hist = before_h[slots[rr]].clone()
            meta = torch.zeros(3, dtype=torch.int64)
            cand = torch.empty(m + 1, 3, dtype=torch.int64)
            up_e._hash_request[(1,)](ids[s0:s1].contiguous(), positions[s0:s1].contiguous(), token_map,
                                     valid[s0:s1].contiguous(), hist, meta, N=m, VOCAB=VOCAB,
                                     MASKED=True, MAX_CONTEXT=1 << 20, BLOCK=triton.next_power_of_2(m))
            up_e._history_candidates[(m + 1,)](ids[s0:s1].contiguous(), positions[s0:s1].contiguous(),
                                               valid[s0:s1].contiguous(), token_map, hist, cand,
                                               N=m, VOCAB=VOCAB, MASKED=True)
            up_e._commit_history[(1,)](cand, meta, hist, counts[rr : rr + 1].long())
            assert torch.equal(hs.history[slots[rr]], hist), ("spec history", trial, rr)
            # upstream C2, this request alone
            part = before_p[slots[rr]].clone()
            ccand = torch.zeros(7, 2, 512)
            cmeta = torch.zeros(3, dtype=torch.int64)
            position = torch.tensor([starts[rr]], dtype=torch.int64)
            out = torch.empty(m, 512, dtype=torch.float16)
            complete = torch.empty(m, dtype=torch.bool)
            up_c._begin[(1,)](positions[s0:s1].contiguous(), position, cmeta, N=m, BLOCK=triton.next_power_of_2(m))
            up_c._pool_c2[(m,)](kv[s0:s1].contiguous(), score[s0:s1].contiguous(), positions[s0:s1].contiguous(),
                                part, ccand, norm, out, complete, N=m, D=512, EPS=1e-20, BLOCK=512)
            up_c._commit_c2[(1,)](part, ccand, cmeta, position, counts[rr : rr + 1].long(), D=512, BLOCK=512)
            assert torch.equal(c2.partial[slots[rr]], part), ("spec partial", trial, rr)
            assert int(c2.committed[slots[rr]]) == int(position), ("spec position", trial, rr)
        for s in set(range(S + 1)) - set(slots):
            assert torch.equal(hs.history[s], before_h[s]) and torch.equal(c2.partial[s], before_p[s])
    # a prefill chunk (n > K) must commit all of its inputs
    hs = st.MuEngramHashState(token_map, primes, offsets, multipliers, slots=S, max_tokens=64,
                              max_context=4096, trap=False)
    c2 = st.MuC2State(norm, slots=S, trap=False)
    spec = st.MuSpeculation(hs, [c2], slots=S, max_inputs=K)
    qsl, positions, valid, req_slot = make_batch([9], [0], [1])
    token_req = st.build_token_req(qsl, 1, 9, torch.empty(9, dtype=torch.int32))
    spec.snapshot((torch.randint(0, VOCAB, (9,)), positions, valid, [(torch.randn(9, 512), torch.randn(9, 512))]),
                  qsl, req_slot, token_req, 1)
    spec.commit(req_slot, torch.tensor([4], dtype=torch.int32), 1)
    assert int(hs.err) & st.ERR_CONTINUE and int(c2.err) & st.ERR_CONTINUE
    print("spec OK")

print("ALL OK")

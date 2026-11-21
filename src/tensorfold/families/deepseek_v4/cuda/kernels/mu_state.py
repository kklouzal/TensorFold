# SPDX-License-Identifier: Apache-2.0
# Provenance: friend overlay patch 85 (SPDX Apache-2.0), snapshot lp-snap-c1es8-numa-admit2-imgtext-share2-fix5-ram4-stepbud-ram3-memfix-ctx1m-ppovl-opt2-pfx-1-eng-pfxfix4,
#   patches/ds41mu_files/local_mu_state.py (sha256 acbaca1a13b778afff9628ed35bf35f703cc97d6b6ba6712de291f82437ad936).
# TensorFold changes (kernel arithmetic verbatim):
# - @triton.jit -> triton_utils.jit: the same JITFunction; launches add the SM70 Triton 3.2 profile options
#   (ptx_version=83, needed with CUDA 12.9 ptxas). Kernel arithmetic unchanged.
# - Local interpreter/GPU switch replaced by the package's: _exp = triton_utils.tldevice.exp (libdevice exp when
#   compiled, identical to the overlay; tl.exp under TRITON_INTERPRET) and _trap = shared_protocol._fail_if (the
#   same PTX trap when compiled; under the interpreter it raises instead of the overlay's no-op, so trap=True
#   validation failures surface on CPU too; trap=False keeps reporting through `err` only).
# - Triton 3.2: no rewrite was needed.
"""LOCAL-PATCH 85 (ds41_mu) state module.

Multi-request DeepSeek-V4.1 per-sequence state, indexed by request slot.

Upstream (deepseek_v41_engram.py V41EngramHashState, deepseek_v41_compression.py
V41C2State) keeps ONE sequence: a [4] Engram history (3 compressed token ids +
committed position) and a [2, 512] FP32 C2 partial group per ratio-2 source,
and validates that the forward is one contiguous token chain.

Here the same state has a slot axis ([S + 1, ...], row S is a scratch row for
padding/dummy requests) and every kernel resolves, per token, its request r
(token_req, built from query_start_loc on the device), the request's first
token in the batch (query_start_loc[r]) and its slot (req_slot[r]). The
arithmetic per token is the upstream kernels' arithmetic, unchanged, so a
request computed inside a batch gives the same bits as computed alone, and a
token never reads another request's tokens or state:

  * Engram hash: lag `at = local - lag`; in-batch only while at >= 0 (the
    request's own tokens), else the slot's history; history ignored at
    position 0 (fresh sequence).
  * C2 pooling: the previous token is in-batch only when local > 0, else the
    slot's partial row.
  * commit (no drafts: every scheduled input is accepted): per request, the
    last three compressed ids / the last token's partial group, and the new
    committed position.

Validation (error word, trap on GPU): token ids in range, positions
contiguous inside each request, a continuing request (start > 0) starts at
its slot's committed position, start + length <= max context.

Pure Triton; no vLLM import (tests run it under TRITON_INTERPRET=1).
Torch references (`*_reference`) restate the semantics loop by loop.
"""

import torch

from .shared_protocol import _fail_if as _trap
from .triton_utils import jit, tl, tldevice, triton

ERR_TOKEN = 1  # token id outside the vocabulary
ERR_POSITION = 2  # positions not contiguous inside a request
ERR_CONTINUE = 4  # continuing request does not start at the committed position
ERR_CONTEXT = 8  # start + length beyond the state's max context
ERR_SLOT = 16  # slot outside [0, S]

@jit
def _exp(x):
    return tldevice.exp(x)


@jit
def _report(err, bits, TRAP: tl.constexpr):
    if bits != 0:
        tl.atomic_or(err, bits.to(tl.int32))
        if TRAP:
            _trap(bits)


# ---------------------------------------------------------------- batch map
@jit(do_not_specialize=["T", "R"])
def _token_req(qsl, out, T, R, RB: tl.constexpr, BLOCK: tl.constexpr):
    """out[t] = r with qsl[r] <= t < qsl[r + 1], or -1 past qsl[R] (padding)."""
    t = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    j = tl.arange(0, RB)
    ends = tl.load(qsl + 1 + j, j < R, 2147483647)
    r = tl.sum((ends[None, :] <= t[:, None]).to(tl.int32), 1)
    total = tl.load(qsl + R)
    tl.store(out + t, tl.where(t < total, r, -1), t < T)


def build_token_req(query_start_loc, num_reqs, num_tokens, out, rb=None):
    """Per-token request index for the first num_tokens rows (graph-safe).

    rb: a fixed power of two >= num_reqs (one compiled kernel for all batch
    sizes); default next_power_of_2(num_reqs)."""
    assert query_start_loc.dtype == torch.int32 and out.dtype == torch.int32
    assert query_start_loc.numel() >= num_reqs + 1 and out.numel() >= num_tokens
    rb = rb or triton.next_power_of_2(max(num_reqs, 1))
    assert rb >= num_reqs
    if num_tokens:
        _token_req[(triton.cdiv(num_tokens, 128),)](
            query_start_loc,
            out,
            num_tokens,
            num_reqs,
            RB=rb,
            BLOCK=128,
        )
    return out[:num_tokens]


def token_req_reference(query_start_loc, num_reqs, num_tokens):
    qsl = query_start_loc[: num_reqs + 1].tolist()
    out = torch.full((num_tokens,), -1, dtype=torch.int32)
    for r in range(num_reqs):
        out[qsl[r] : qsl[r + 1]] = r
    return out


# ------------------------------------------------------------ Engram hashes
@jit
def _mu_compressed(ids, valid, token_map, s0, at, inside, VOCAB: tl.constexpr):
    token = tl.load(ids + s0 + at, inside, 0).to(tl.int64)
    token = tl.where((token >= 0) & (token < VOCAB), token, 0)
    compressed = tl.load(token_map + token, inside, 0).to(tl.int64)
    live = tl.load(valid + s0 + at, inside, 0) != 0
    return tl.where(live, compressed, -1)


@jit
def _mu_hash_rows(
    ids,
    positions,
    valid,
    token_req,
    qsl,
    req_slot,
    token_map,
    primes,
    offsets,
    multipliers,
    pad,
    history,
    hashes,
    err,
    S: tl.constexpr,
    VOCAB: tl.constexpr,
    MAX_CONTEXT: tl.constexpr,
    TRAP: tl.constexpr,
):
    t, layer = tl.program_id(0), tl.program_id(1)
    heads = tl.arange(0, 32)
    r = tl.load(token_req + t)
    rr = tl.where(r >= 0, r, 0)
    s0 = tl.load(qsl + rr).to(tl.int64)
    s1 = tl.load(qsl + rr + 1).to(tl.int64)
    mine = (r >= 0) & (t >= s0) & (t < s1)
    slot = tl.load(req_slot + rr, mine, S).to(tl.int64)
    bad_slot = mine & ((slot < 0) | (slot > S))
    slot = tl.where(bad_slot | ~mine, S, slot)
    local = tl.where(mine, t - s0, 0)
    start = tl.load(positions + s0, mine, 0).to(tl.int64)
    blocked = tl.full((), False, tl.int1)
    products = tl.full((4,), 0, tl.int64)
    for lag in tl.static_range(4):
        at = local - lag
        inside = mine & (at >= 0)
        current = _mu_compressed(ids, valid, token_map, s0, at, inside, VOCAB)
        old = tl.load(
            history + slot * 4 + at + 3, mine & (at < 0) & (at >= -3), -1
        )
        old = tl.where(start == 0, -1, old)
        current = tl.where(at >= 0, current, old)
        blocked |= current == -1
        current = tl.where(blocked, tl.load(pad), current)
        product = current * tl.load(multipliers + layer * 4 + lag)
        products = tl.where(tl.arange(0, 4) == lag, product, products)
    p0 = tl.sum(tl.where(tl.arange(0, 4) == 0, products, 0), 0)
    p1 = tl.sum(tl.where(tl.arange(0, 4) == 1, products, 0), 0)
    p2 = tl.sum(tl.where(tl.arange(0, 4) == 2, products, 0), 0)
    p3 = tl.sum(tl.where(tl.arange(0, 4) == 3, products, 0), 0)
    value = p0 ^ p1
    value = tl.where(heads >= 8, value ^ p2, value)
    value = tl.where(heads >= 16, value ^ p3, value)
    prime = tl.load(primes + layer * 24 + heads, heads < 24, 1)
    base = tl.load(offsets + layer * 24 + heads, heads < 24, 0)
    tl.store(hashes + (t * 2 + layer) * 24 + heads, value % prime + base, heads < 24)
    if layer == 0:
        # Validation once per token.
        live = mine & (tl.load(valid + t) != 0)
        token = tl.load(ids + t).to(tl.int64)
        bits = tl.where(live & ((token < 0) | (token >= VOCAB)), 1, 0)
        position = tl.load(positions + t).to(tl.int64)
        bits |= tl.where(live & (position != start + local), 2, 0)
        committed = tl.load(history + slot * 4 + 3)
        bits |= tl.where(
            mine & (slot < S) & (local == 0) & (start != 0) & (start != committed), 4, 0
        )
        # Scratch-slot rows (padding, dummy batches) are never validated.
        bits |= tl.where(
            mine & (slot < S) & (local == 0) & (start + (s1 - s0) > MAX_CONTEXT), 8, 0
        )
        bits |= tl.where(bad_slot, 16, 0)
        _report(err, bits, TRAP)


@jit
def _mu_commit_history(
    ids,
    positions,
    valid,
    qsl,
    req_slot,
    token_req,
    token_map,
    history,
    S: tl.constexpr,
    VOCAB: tl.constexpr,
):
    """No drafts: every scheduled input of request r is accepted.

    A request commits only if its first token maps to it in token_req: dummy
    batches (token_req all -1, see build_metadata_view) never touch a slot,
    even when req_slot still holds the last real step's slots."""
    r = tl.program_id(0)
    s0 = tl.load(qsl + r).to(tl.int64)
    n = tl.load(qsl + r + 1).to(tl.int64) - s0
    slot = tl.load(req_slot + r).to(tl.int64)
    owner = tl.load(token_req + s0, n > 0, -1)
    if (n > 0) & (slot >= 0) & (slot < S) & (owner == r):
        start = tl.load(positions + s0).to(tl.int64)
        col = tl.arange(0, 4)
        at = n - 3 + col
        inside = (col < 3) & (at >= 0)
        token = tl.load(ids + s0 + at, inside, 0).to(tl.int64)
        token = tl.where((token >= 0) & (token < VOCAB), token, 0)
        compressed = tl.load(token_map + token, inside, 0).to(tl.int64)
        live = tl.load(valid + s0 + at, inside, 0) != 0
        compressed = tl.where(live, compressed, -1)
        old = tl.load(history + slot * 4 + at + 3, (col < 3) & (at < 0), -1)
        old = tl.where(start == 0, -1, old)
        value = tl.where(at >= 0, compressed, old)
        value = tl.where(col < 3, value, start + n)
        tl.store(history + slot * 4 + col, value)


class MuEngramHashState(torch.nn.Module):
    """V41EngramHashState with a slot axis: history [S + 1, 4] int64."""

    def __init__(
        self,
        token_map,
        primes,
        offsets,
        multipliers,
        *,
        slots,
        pad_token_id=2,
        max_tokens=2048,
        max_context=16384,
        trap=True,
    ):
        super().__init__()
        device = token_map.device
        if (
            token_map.ndim != 1
            or primes.shape != (2, 24)
            or offsets.shape != (2, 24)
            or multipliers.shape != (2, 4)
            or not 1 <= slots <= 4096
            or not 1 <= max_tokens <= max_context <= 1048576
        ):
            raise ValueError("LOCAL-PATCH 85: invalid batched Engram hash state")
        for name, value in (
            ("token_map", token_map),
            ("primes", primes),
            ("offsets", offsets),
            ("multipliers", multipliers),
        ):
            self.register_buffer(name, value.contiguous(), persistent=False)
        self.register_buffer(
            "pad", token_map[pad_token_id : pad_token_id + 1].long(), persistent=False
        )
        history = torch.full((slots + 1, 4), -1, dtype=torch.int64, device=device)
        history[:, 3] = 0
        self.register_buffer("history", history, persistent=False)
        self.register_buffer(
            "err", torch.zeros(1, dtype=torch.int32, device=device), persistent=False
        )
        self.slots, self.max_tokens, self.max_context = slots, max_tokens, max_context
        self.trap = trap

    def forward(self, input_ids, positions, valid, token_req, qsl, req_slot):
        n = input_ids.numel()
        if not 0 < n <= self.max_tokens or positions.shape != input_ids.shape:
            raise ValueError("LOCAL-PATCH 85: Engram hash batch shape")
        result = torch.empty(n, 2, 24, dtype=torch.int64, device=input_ids.device)
        _mu_hash_rows[(n, 2)](
            input_ids,
            positions,
            valid,
            token_req,
            qsl,
            req_slot,
            self.token_map,
            self.primes,
            self.offsets,
            self.multipliers,
            self.pad,
            self.history,
            result,
            self.err,
            S=self.slots,
            VOCAB=self.token_map.numel(),
            MAX_CONTEXT=self.max_context,
            TRAP=self.trap,
        )
        return result

    def commit_all(self, input_ids, positions, valid, qsl, req_slot, num_reqs, token_req):
        if num_reqs:
            _mu_commit_history[(num_reqs,)](
                input_ids,
                positions,
                valid,
                qsl,
                req_slot,
                token_req,
                self.token_map,
                self.history,
                S=self.slots,
                VOCAB=self.token_map.numel(),
            )

    def reset_slot(self, slot):
        self.history[slot].copy_(
            torch.tensor([-1, -1, -1, 0], dtype=torch.int64, device=self.history.device)
        )


def hash_reference(
    ids, positions, valid, qsl, num_reqs, req_slot, history, token_map, primes,
    offsets, multipliers, pad,
):
    """Per request, the single-sequence _hash_rows semantics (history per slot)."""
    n = ids.numel()
    out = torch.zeros(n, 2, 24, dtype=torch.int64)
    q = qsl.tolist()
    for r in range(num_reqs):
        s0, s1 = q[r], q[r + 1]
        slot = int(req_slot[r])
        start = int(positions[s0]) if s1 > s0 else 0
        for t in range(s0, s1):
            local = t - s0
            for layer in range(2):
                blocked = False
                products = []
                for lag in range(4):
                    at = local - lag
                    if at >= 0:
                        cur = (
                            int(token_map[int(ids[s0 + at])])
                            if bool(valid[s0 + at])
                            else -1
                        )
                    elif at >= -3 and start != 0:
                        cur = int(history[slot, at + 3])
                    else:
                        cur = -1
                    blocked |= cur == -1
                    if blocked:
                        cur = int(pad)
                    products.append(cur * int(multipliers[layer, lag]))
                for h in range(24):
                    v = products[0] ^ products[1]
                    if h >= 8:
                        v ^= products[2]
                    if h >= 16:
                        v ^= products[3]
                    p = int(primes[layer, h])
                    out[t, layer, h] = (v % p) + int(offsets[layer, h])
    return out


def commit_history_reference(ids, positions, valid, qsl, num_reqs, req_slot, history, token_map):
    history = history.clone()
    q = qsl.tolist()
    for r in range(num_reqs):
        s0, s1 = q[r], q[r + 1]
        n = s1 - s0
        if n <= 0:
            continue
        slot = int(req_slot[r])
        start = int(positions[s0])
        new = []
        for col in range(3):
            at = n - 3 + col
            if at >= 0:
                new.append(int(token_map[int(ids[s0 + at])]) if bool(valid[s0 + at]) else -1)
            else:
                new.append(int(history[slot, at + 3]) if start != 0 else -1)
        history[slot] = torch.tensor(new + [start + n])
    return history


# -------------------------------------------------------------- C2 pooling
@jit
def _mu_pool_c2(
    kv,
    score,
    positions,
    token_req,
    qsl,
    req_slot,
    partial,
    committed,
    norm,
    out,
    complete,
    err,
    S: tl.constexpr,
    D: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
    TRAP: tl.constexpr,
):
    t = tl.program_id(0)
    d = tl.arange(0, BLOCK)
    valid = d < D
    r = tl.load(token_req + t)
    rr = tl.where(r >= 0, r, 0)
    s0 = tl.load(qsl + rr).to(tl.int64)
    s1 = tl.load(qsl + rr + 1).to(tl.int64)
    mine = (r >= 0) & (t >= s0) & (t < s1)
    slot = tl.load(req_slot + rr, mine, S).to(tl.int64)
    slot = tl.where(mine & (slot >= 0) & (slot <= S), slot, S)
    local = tl.where(mine, t - s0, 0)
    position = tl.load(positions + t).to(tl.int64)
    current = tl.load(kv + t * D + d, valid, 0)
    current_score = tl.load(score + t * D + d, valid, 0)
    inner = mine & (local > 0)
    previous = tl.load(kv + (t - 1) * D + d, inner & valid, 0)
    previous_score = tl.load(score + (t - 1) * D + d, inner & valid, 0)
    previous = tl.where(inner, previous, tl.load(partial + slot * 2 * D + d, valid, 0))
    previous_score = tl.where(
        inner, previous_score, tl.load(partial + (slot * 2 + 1) * D + d, valid, 0)
    )
    done = mine & ((position + 1) % 2 == 0)
    maximum = tl.maximum(previous_score, current_score)
    a = _exp(previous_score - maximum)
    b = _exp(current_score - maximum)
    value = (
        (tl.div_rn(a, a + b) * previous + tl.div_rn(b, a + b) * current)
        .to(tl.float16)
        .to(tl.float32)
    )
    value = tl.where(done, value, 0)
    variance = tl.sum(tl.where(valid, value * value, 0), 0) / D
    value = value * tl.rsqrt(variance + EPS) * tl.load(norm + d, valid, 0)
    tl.store(out + t * D + d, value, valid)
    tl.store(complete + t, done)
    # A continuing request whose first token closes a group reads the slot's
    # partial row: it must be the row committed at exactly this position.
    stale = mine & (slot < S) & (local == 0) & done & (tl.load(committed + slot) != position)
    _report(err, tl.where(stale, 4, 0), TRAP)


@jit
def _mu_commit_c2(
    kv,
    score,
    positions,
    qsl,
    req_slot,
    token_req,
    partial,
    committed,
    S: tl.constexpr,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
):
    r = tl.program_id(0)
    s0 = tl.load(qsl + r).to(tl.int64)
    s1 = tl.load(qsl + r + 1).to(tl.int64)
    slot = tl.load(req_slot + r).to(tl.int64)
    owner = tl.load(token_req + s0, s1 > s0, -1)  # -1: dummy batch (no commit)
    if (s1 > s0) & (slot >= 0) & (slot < S) & (owner == r):
        last = s1 - 1
        position = tl.load(positions + last).to(tl.int64)
        done = (position + 1) % 2 == 0
        d = tl.arange(0, BLOCK)
        valid = d < D
        value = tl.load(kv + last * D + d, valid, 0)
        value_score = tl.load(score + last * D + d, valid, 0)
        tl.store(partial + slot * 2 * D + d, tl.where(done, 0, value), valid)
        tl.store(partial + (slot * 2 + 1) * D + d, tl.where(done, 0, value_score), valid)
        tl.store(committed + slot, position + 1)


class MuC2State(torch.nn.Module):
    """V41C2State with a slot axis: partial [S + 1, 2, 512] FP32."""

    def __init__(self, norm_weight, *, slots, eps=1e-20, trap=True):
        super().__init__()
        if norm_weight.dtype != torch.float32 or norm_weight.shape != (512,):
            raise ValueError("LOCAL-PATCH 85: C2 norm must be FP32 width 512")
        device = norm_weight.device
        self.register_buffer("norm_weight", norm_weight, persistent=False)
        self.register_buffer(
            "partial",
            torch.zeros(slots + 1, 2, 512, dtype=torch.float32, device=device),
            persistent=False,
        )
        self.register_buffer(
            "committed",
            torch.zeros(slots + 1, dtype=torch.int64, device=device),
            persistent=False,
        )
        self.register_buffer(
            "err", torch.zeros(1, dtype=torch.int32, device=device), persistent=False
        )
        self.slots, self.eps, self.trap = slots, eps, trap

    def forward(self, kv, score, positions, token_req, qsl, req_slot):
        n = kv.shape[0]
        if (
            kv.shape != score.shape
            or kv.shape[1:] != (512,)
            or kv.dtype != torch.float32
            or score.dtype != torch.float32
            or positions.shape != (n,)
        ):
            raise ValueError("LOCAL-PATCH 85: C2 requires FP32 projections")
        result = torch.empty_like(kv, dtype=torch.float16)
        complete = torch.empty(n, dtype=torch.bool, device=kv.device)
        if n:
            _mu_pool_c2[(n,)](
                kv.contiguous(),
                score.contiguous(),
                positions,
                token_req,
                qsl,
                req_slot,
                self.partial,
                self.committed,
                self.norm_weight,
                result,
                complete,
                self.err,
                S=self.slots,
                D=512,
                EPS=self.eps,
                BLOCK=512,
                TRAP=self.trap,
                enable_fp_fusion=False,
            )
        return result, complete

    def commit_all(self, kv, score, positions, qsl, req_slot, num_reqs, token_req):
        if num_reqs:
            _mu_commit_c2[(num_reqs,)](
                kv.contiguous(),
                score.contiguous(),
                positions,
                qsl,
                req_slot,
                token_req,
                self.partial,
                self.committed,
                S=self.slots,
                D=512,
                BLOCK=512,
            )

    def reset_slot(self, slot):
        self.partial[slot].zero_()
        self.committed[slot] = 0


def pool_c2_reference(kv, score, positions, qsl, num_reqs, req_slot, partial, norm, eps=1e-20):
    """FP32 torch restatement (exact rounding boundaries; exp may differ by ulp)."""
    out = torch.zeros(kv.shape[0], 512, dtype=torch.float16)
    q = qsl.tolist()
    for r in range(num_reqs):
        slot = int(req_slot[r])
        for t in range(q[r], q[r + 1]):
            if t > q[r]:
                prev, prev_s = kv[t - 1], score[t - 1]
            else:
                prev, prev_s = partial[slot, 0], partial[slot, 1]
            if (int(positions[t]) + 1) % 2:
                continue
            m = torch.maximum(prev_s, score[t])
            a, b = torch.exp(prev_s - m), torch.exp(score[t] - m)
            v = (a / (a + b) * prev + b / (a + b) * kv[t]).half().float()
            v = v * torch.rsqrt((v * v).sum() / 512 + eps) * norm
            out[t] = v.half()
    return out


# ------------------------------------------- speculation (phase 4, DSpark)
# Per-slot candidate snapshots and commit by accepted count: the upstream
# single-sequence _history_candidates / _commit_history and _pool_c2's
# candidate rows / _commit_c2, with a slot axis. Not used by the forward yet
# (patch 85 admits no speculative decoding); the DSpark integration snapshots
# in the forward and commits each request's accepted count from the runner.
# Rows: j = number of committed inputs (0..n). A request with n >= J - 1
# inputs (a prefill chunk) keeps only j = n, in row J - 1, and must commit it.
@jit
def _mu_history_candidates(
    ids, positions, valid, qsl, req_slot, token_req, token_map, history, cand,
    S: tl.constexpr, VOCAB: tl.constexpr, J: tl.constexpr,
):
    r, j = tl.program_id(0), tl.program_id(1)
    s0 = tl.load(qsl + r).to(tl.int64)
    n = tl.load(qsl + r + 1).to(tl.int64) - s0
    slot = tl.load(req_slot + r).to(tl.int64)
    owner = tl.load(token_req + s0, n > 0, -1)
    full = n >= J - 1
    count = tl.where(full, n, j)
    keep = (n > 0) & (slot >= 0) & (slot < S) & (owner == r) & (count <= n)
    keep &= (~full) | (j == J - 1)
    if keep:
        start = tl.load(positions + s0).to(tl.int64)
        col = tl.arange(0, 4)
        at = count - 3 + col
        inside = (col < 3) & (at >= 0)
        token = tl.load(ids + s0 + at, inside, 0).to(tl.int64)
        token = tl.where((token >= 0) & (token < VOCAB), token, 0)
        compressed = tl.load(token_map + token, inside, 0).to(tl.int64)
        live = tl.load(valid + s0 + at, inside, 0) != 0
        compressed = tl.where(live, compressed, -1)
        old = tl.load(history + slot * 4 + at + 3, (col < 3) & (at < 0), -1)
        old = tl.where(start == 0, -1, old)
        value = tl.where(at >= 0, compressed, old)
        value = tl.where(col < 3, value, start + count)
        tl.store(cand + (slot * J + j) * 4 + col, value)


@jit
def _mu_commit_history_accepted(history, cand, cand_n, req_slot, counts, err, S: tl.constexpr, J: tl.constexpr):
    """history[slot] = candidate of `counts[r]` inputs (snapshot of this step)."""
    r = tl.program_id(0)
    slot = tl.load(req_slot + r).to(tl.int64)
    count = tl.load(counts + r).to(tl.int64)
    if (slot >= 0) & (slot < S) & (count >= 0):
        n = tl.load(cand_n + slot)
        row = tl.where(n >= J - 1, J - 1, count)
        bad = (count > n) | ((n >= J - 1) & (count != n))
        col = tl.arange(0, 4)
        value = tl.load(cand + (slot * J + row) * 4 + col)
        tl.store(history + slot * 4 + col, value, ~bad)
        tl.atomic_or(err, tl.where(bad, 4, 0).to(tl.int32))


@jit
def _mu_snapshot_lengths(qsl, req_slot, token_req, cand_n, S: tl.constexpr):
    r = tl.program_id(0)
    s0 = tl.load(qsl + r).to(tl.int64)
    n = tl.load(qsl + r + 1).to(tl.int64) - s0
    slot = tl.load(req_slot + r).to(tl.int64)
    owner = tl.load(token_req + s0, n > 0, -1)
    if (n > 0) & (slot >= 0) & (slot < S) & (owner == r):
        tl.store(cand_n + slot, n)


@jit
def _mu_c2_candidates(
    kv, score, positions, qsl, req_slot, token_req, partial, cand, cand_pos,
    S: tl.constexpr, J: tl.constexpr, D: tl.constexpr, BLOCK: tl.constexpr,
):
    r, j = tl.program_id(0), tl.program_id(1)
    s0 = tl.load(qsl + r).to(tl.int64)
    n = tl.load(qsl + r + 1).to(tl.int64) - s0
    slot = tl.load(req_slot + r).to(tl.int64)
    owner = tl.load(token_req + s0, n > 0, -1)
    full = n >= J - 1
    count = tl.where(full, n, j)
    keep = (n > 0) & (slot >= 0) & (slot < S) & (owner == r) & (count <= n)
    keep &= (~full) | (j == J - 1)
    if keep:
        start = tl.load(positions + s0).to(tl.int64)
        d = tl.arange(0, BLOCK)
        valid = d < D
        last = s0 + count - 1
        done = (start + count) % 2 == 0  # the last committed input closes its group
        current = tl.load(kv + last * D + d, valid & (count > 0), 0)
        current_score = tl.load(score + last * D + d, valid & (count > 0), 0)
        old = tl.load(partial + slot * 2 * D + d, valid, 0)
        old_score = tl.load(partial + (slot * 2 + 1) * D + d, valid, 0)
        first = tl.where(start == 0, 0, old)
        first_score = tl.where(start == 0, 0, old_score)
        value = tl.where(count > 0, tl.where(done, 0, current), first)
        value_score = tl.where(count > 0, tl.where(done, 0, current_score), first_score)
        tl.store(cand + ((slot * J + j) * 2) * D + d, value, valid)
        tl.store(cand + ((slot * J + j) * 2 + 1) * D + d, value_score, valid)
        tl.store(cand_pos + slot * J + j, start + count)


@jit
def _mu_commit_c2_accepted(
    partial, committed, cand, cand_pos, cand_n, req_slot, counts, err,
    S: tl.constexpr, J: tl.constexpr, D: tl.constexpr, BLOCK: tl.constexpr,
):
    r = tl.program_id(0)
    slot = tl.load(req_slot + r).to(tl.int64)
    count = tl.load(counts + r).to(tl.int64)
    if (slot >= 0) & (slot < S) & (count >= 0):
        n = tl.load(cand_n + slot)
        row = tl.where(n >= J - 1, J - 1, count)
        bad = (count > n) | ((n >= J - 1) & (count != n))
        d = tl.arange(0, BLOCK)
        valid = (d < D) & ~bad
        for part in tl.static_range(2):
            value = tl.load(cand + ((slot * J + row) * 2 + part) * D + d, d < D, 0)
            tl.store(partial + (slot * 2 + part) * D + d, value, valid)
        tl.store(committed + slot, tl.load(cand_pos + slot * J + row), ~bad)
        tl.atomic_or(err, tl.where(bad, 4, 0).to(tl.int32))


class MuSpeculation:
    """Per-slot candidate snapshots for one Engram hash state and its C2 states.

    snapshot() runs in the forward (after the hash / pooling kernels read the
    slot state, instead of commit_all); commit(counts) runs from the runner
    with the number of inputs accepted per batch row (prefill rows: all)."""

    def __init__(self, hash_state, c2_states, *, slots, max_inputs):
        device = (hash_state.history if hash_state is not None else c2_states[0].partial).device
        self.hash_state, self.c2_states = hash_state, list(c2_states)
        self.slots, self.J = slots, max_inputs + 2
        self.cand_n = torch.zeros(slots + 1, dtype=torch.int64, device=device)
        self.history = (
            torch.full((slots + 1, self.J, 4), -1, dtype=torch.int64, device=device)
            if hash_state is not None
            else None
        )
        self.partials = [
            torch.zeros(slots + 1, self.J, 2, 512, dtype=torch.float32, device=device)
            for _ in self.c2_states
        ]
        self.positions = [
            torch.zeros(slots + 1, self.J, dtype=torch.int64, device=device)
            for _ in self.c2_states
        ]

    def snapshot(self, step_inputs, qsl, req_slot, token_req, num_reqs):
        """step_inputs: (ids, positions, valid) and per C2 state (kv, score)."""
        ids, positions, valid, c2_inputs = step_inputs
        if not num_reqs:
            return
        _mu_snapshot_lengths[(num_reqs,)](qsl, req_slot, token_req, self.cand_n, S=self.slots)
        if self.hash_state is not None:
            _mu_history_candidates[(num_reqs, self.J)](
                ids, positions, valid, qsl, req_slot, token_req,
                self.hash_state.token_map, self.hash_state.history, self.history,
                S=self.slots, VOCAB=self.hash_state.token_map.numel(), J=self.J,
            )
        for state, cand, pos, (kv, score) in zip(self.c2_states, self.partials, self.positions, c2_inputs):
            _mu_c2_candidates[(num_reqs, self.J)](
                kv.contiguous(), score.contiguous(), positions, qsl, req_slot, token_req,
                state.partial, cand, pos, S=self.slots, J=self.J, D=512, BLOCK=512,
            )

    def commit(self, req_slot, counts, num_reqs):
        if not num_reqs:
            return
        if self.hash_state is not None:
            _mu_commit_history_accepted[(num_reqs,)](
                self.hash_state.history, self.history, self.cand_n, req_slot, counts,
                self.hash_state.err, S=self.slots, J=self.J,
            )
        for state, cand, pos in zip(self.c2_states, self.partials, self.positions):
            _mu_commit_c2_accepted[(num_reqs,)](
                state.partial, state.committed, cand, pos, self.cand_n, req_slot, counts,
                state.err, S=self.slots, J=self.J, D=512, BLOCK=512,
            )

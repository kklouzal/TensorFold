"""Gemma4 CUDA attention over persistent caches: float64 oracle, row-count invariance, cache lifecycle.

The oracle builds Q/K/V with the unfused norm and RoPE kernels (the previous
forward's exact tensors) and attends in float64. Production rows must stay
within BF16 output/probability rounding of it, and every forward split of the
same positions (serial, verify windows, prefill chunks, sequential or parallel
full-attention launches) must produce identical bits.
"""

import random

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
if not torch.cuda.is_available():
    pytest.skip("Gemma4 attention kernels need CUDA", allow_module_level=True)

from tensorfold.families.gemma4.cuda import attention  # noqa: E402
from tensorfold.families.gemma4.cuda.forward import apply_rope, ct_rms_norm, rms_norm  # noqa: E402
from tensorfold.families.gemma4.cuda.state import State, capacity_bytes, ring_slots, sliding_slots  # noqa: E402
from tensorfold.families.gemma4.cuda.weights import Config  # noqa: E402

DEV = torch.device("cuda")
EPS = 1e-6


def config(window=1024, layers=("sliding_attention", "full_attention")):
    return Config(hidden=2816, layers=len(layers), vocab=262144, heads=16, kv_heads=8, global_kv_heads=2,
                  head_dim=256, global_head_dim=512, intermediate=2112, moe_width=704, experts=128, top_k=8,
                  window=window, max_position=262144, eps=EPS, softcap=30.0, layer_types=tuple(layers),
                  rope_theta=1e4, global_rope_theta=1e6, global_rotary_fraction=0.25, vision_bidirectional=True)


def inverse(c, sliding):
    d = c.head_dim if sliding else c.global_head_dim
    rotated = d if sliding else int(d * c.global_rotary_fraction)
    inv = torch.pow(c.rope_theta if sliding else c.global_rope_theta,
                    -torch.arange(0, d, 2, dtype=torch.float32, device=DEV) / d)
    inv[rotated // 2:] = 0
    return inv


class Stream:
    """Random raw projections for positions 0..T-1, a fixed function of position."""

    def __init__(self, c, total, seed=0, scale=3.0):
        g = torch.Generator(device="cpu").manual_seed(seed)
        h, hk, d, hg, dg = c.heads, c.kv_heads, c.head_dim, c.global_kv_heads, c.global_head_dim

        def rnd(*shape):
            return (torch.randn(*shape, generator=g) * scale).to(torch.bfloat16).to(DEV)

        self.q, self.k, self.v = rnd(total, h * d), rnd(total, hk * d), rnd(total, hk * d)
        self.gq, self.gk = rnd(total, h * dg), rnd(total, hg * dg)
        self.qw, self.kw = rnd(d) * 0.3 + 1, rnd(d) * 0.3 + 1
        self.gqw, self.gkw = rnd(dg) * 0.3 + 1, rnd(dg) * 0.3 + 1


def run(c, s, splits, *, sequential=None, blocks=(), mlx=False, state=None):
    """Production kernels over the given forward sizes -> (sliding out, full out) per position."""
    rows_max = max(splits)
    st = state or State(c, DEV, 4096, rows=max(rows_max, 16))
    sl, fl = [], []
    inv_s, inv_g = inverse(c, True), inverse(c, False)
    for rows in splits:
        p = st.position
        st.check(rows)
        st.pos.fill_(p)
        starts = [-1] * rows
        ends = [-1] * rows
        longest = 0
        for b, e in blocks:
            if p <= b and e <= p + rows:
                longest = max(longest, e - b)
                for r in range(b - p, e - p):
                    starts[r], ends[r] = b, e
        bs = torch.tensor(starts, dtype=torch.int32, device=DEV)
        be = torch.tensor(ends, dtype=torch.int32, device=DEV)
        q = attention.prep_sliding(s.q[p:p + rows], s.k[p:p + rows], s.v[p:p + rows], s.qw, s.kw, inv_s, st.pos,
                                   st.keys[0], st.values[0], eps=EPS, round_normalized=mlx)
        slots = sliding_slots(c.window, rows, c.heads // c.kv_heads, longest)
        sl.append(attention.sliding(q, st.keys[0], st.values[0], st.pos, bs, be, window=c.window, slots=slots))
        q = attention.prep_full(s.gq[p:p + rows], s.gk[p:p + rows], s.gqw, inv_g, st.pos, st.raw[0], st.inv[0],
                                eps=EPS, round_normalized=mlx)
        chunks = (p + rows - 1) // attention.GLOBAL_CHUNK + 1
        fl.append(attention.full(q, st.raw[0], st.inv[0], s.gkw, inv_g, st.pos, chunks=chunks,
                                 round_normalized=mlx, sequential=sequential))
        st.advance(rows)
    return torch.cat(sl), torch.cat(fl), st


def unfused(c, s, total, mlx=False):
    """The previous forward's exact BF16 Q/K/V tensors (unfused norm and RoPE kernels)."""
    norm = (lambda x, w: rms_norm(x, w, EPS)) if mlx else (lambda x, w: ct_rms_norm(x, w, EPS))
    h, hk, d, hg, dg = c.heads, c.kv_heads, c.head_dim, c.global_kv_heads, c.global_head_dim
    inv_s, inv_g = inverse(c, True), inverse(c, False)
    out = {}
    for name, (qr, kr, vr, qw, kw, inv, heads, kvh, dim) in {
            "sliding": (s.q, s.k, s.v, s.qw, s.kw, inv_s, h, hk, d),
            "full": (s.gq, s.gk, s.gk, s.gqw, s.gkw, inv_g, h, hg, dg)}.items():
        q = apply_rope(norm(qr[:total].reshape(total, heads, dim), qw), inv, 0)
        k = apply_rope(norm(kr[:total].reshape(total, kvh, dim), kw), inv, 0)
        v = norm(vr[:total].reshape(total, kvh, dim), None)
        out[name] = q, k, v
    return out


def oracle(q, k, v, window=0, blocks=()):
    """Float64 softmax attention of exact BF16 inputs (scale 1, causal or SWA + visual blocks)."""
    total, h, d = q.shape
    g = h // k.shape[1]
    keys = k.double().repeat_interleave(g, 1).transpose(0, 1)
    values = v.double().repeat_interleave(g, 1).transpose(0, 1)
    pos = torch.arange(total, device=q.device)
    mask = pos[None, :] <= pos[:, None]
    if window:
        for b, e in blocks:
            inside = (pos >= b) & (pos < e)
            mask |= inside[:, None] & inside[None, :]
        mask &= pos[None, :] >= pos[:, None] - window + 1
    scores = torch.einsum("thd,hkd->htk", q.double(), keys).masked_fill(~mask[None], float("-inf"))
    return torch.einsum("htk,hkd->thd", torch.softmax(scores, -1), values)


def within_bf16(actual, expected, values):
    # Output rounding (2^-9) plus BF16 probabilities in P@V (2^-9 each, convex
    # combination): |error| <= 2^-8 max|v| per head, with margin for the fold.
    bound = values.double().abs().amax() * 2.0 ** -7
    err = (actual.double() - expected).abs()
    assert err.max() <= bound, (err.max().item(), bound.item())
    assert err.mean() <= bound / 16


@pytest.mark.parametrize("mlx", [False, True])
def test_production_attention_matches_float64_oracle_through_ring_wrap(mlx):
    c = config(window=64)
    total = 700
    s = Stream(c, total, seed=1)
    sl, fl, _ = run(c, s, [128] * 5 + [60], mlx=mlx)
    ref = unfused(c, s, total, mlx=mlx)
    q, k, v = ref["sliding"]
    within_bf16(sl, oracle(q, k, v, window=64), v)
    q, k, v = ref["full"]
    within_bf16(fl, oracle(q, k, v), v)


def test_fused_prep_publishes_the_unfused_kernels_tensors():
    c = config(window=64)
    total = 200
    s = Stream(c, total, seed=2)
    _, _, st = run(c, s, [100, 100])
    q, k, v = unfused(c, s, total)["sliding"]
    slots = torch.arange(total - st.ring, total, device=DEV) % st.ring
    keep = torch.arange(total - st.ring, total, device=DEV)

    def ulps(a, b):
        return (a.view(torch.int16).int() - b.view(torch.int16).int()).abs().max().item()

    # The reciprocal RMS may differ in its last FP32 bit (reduction order): at most one BF16 ulp.
    assert ulps(st.keys[0][slots], k[keep]) <= 1 and ulps(st.values[0][slots], v[keep]) <= 1
    assert torch.equal(st.raw[0][:total], s.gk[:total].reshape(total, 2, 512))
    expected = s.gk[:total].reshape(total, 2, 512).float().pow(2).mean(-1).add(EPS).rsqrt()
    assert torch.allclose(st.inv[0][:total], expected, rtol=2e-6)


def test_every_forward_split_produces_identical_bits():
    c = config()
    total = 1300                                # crosses the 1024 window, ring wrap and two 512-key chunks
    s = Stream(c, total, seed=3)
    serial = run(c, s, [1] * 40 + [128] * 9 + [108])   # first 40 serial rows, then prefill chunks
    rng = random.Random(4)
    splits = []
    while sum(splits) < total:
        splits.append(min(rng.choice([1, 2, 3, 5, 16, 100, 128]), total - sum(splits)))
    mixed = run(c, s, splits)
    one = run(c, s, [1] * total)
    for a, b in ((serial, one), (mixed, one)):
        assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
    for sequential in (False, True):
        forced = run(c, s, splits, sequential=sequential)
        assert torch.equal(forced[1], one[1])


def test_rollback_and_snapshot_restore_continue_bit_identically():
    c = config(window=64)
    total = 400
    s = Stream(c, total, seed=5)
    reference = run(c, s, [1] * total)
    st = State(c, DEV, 4096, rows=128)
    run(c, s, [128, 100], state=st)
    snap = st.snapshot()
    # A rejected verify window: write 5 rows, keep 2.
    run(c, s, [5], state=st)
    st.rollback(230)
    # A snapshot taken after the rejected window keeps its retained-history bound: the slots of positions
    # 38..40 now hold rejected rows 230..232, so neither the state nor its restored copy may roll back to
    # need them, and the copy continues bit-identically from the earliest admitted rollback.
    after = State(c, DEV, 4096, rows=128)
    after.restore(st.snapshot())
    for state in (st, after):
        with pytest.raises(ValueError, match="sliding-window history"):
            state.rollback(103)
    after.rollback(104)
    again = run(c, s, [8] * 12, state=after)
    assert torch.equal(again[0], reference[0][104:200]) and torch.equal(again[1], reference[1][104:200])
    tail = run(c, s, [5] * 34, state=st)
    assert torch.equal(tail[0], reference[0][230:400]) and torch.equal(tail[1], reference[1][230:400])
    fresh = State(c, DEV, 4096, rows=128)
    fresh.restore(snap)
    resumed = run(c, s, [16] * 10 + [12], state=fresh)
    assert torch.equal(resumed[0], reference[0][228:400]) and torch.equal(resumed[1], reference[1][228:400])
    with pytest.raises(ValueError, match="sliding-window history"):
        st.rollback(100)


def test_visual_blocks_are_bidirectional_in_sliding_layers_only():
    c = config(window=64)
    total = 300
    blocks = ((40, 120), (150, 230))
    s = Stream(c, total, seed=6)
    sl, fl, _ = run(c, s, [40, 80, 30, 80, 70], blocks=blocks)
    ref = unfused(c, s, total)
    q, k, v = ref["sliding"]
    within_bf16(sl, oracle(q, k, v, window=64, blocks=blocks), v)
    q, k, v = ref["full"]
    within_bf16(fl, oracle(q, k, v), v)


def test_visual_prompt_chunking_produces_identical_bits():
    """Prefix-cache independence with images: text rows give the same bits whether or not their forward also
    carries a bidirectional block, and every split that keeps each block whole agrees."""
    c = config(window=64)
    blocks = ((40, 120), (150, 230))
    s = Stream(c, 300, seed=8)
    a = run(c, s, [120, 30, 80, 70], blocks=blocks)
    b = run(c, s, [7, 33, 80, 1, 29, 80, 1, 69], blocks=blocks)
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])


def test_state_bounds_and_capacity_accounting():
    c = config()
    st = State(c, DEV, 8192, rows=128)
    assert st.ring == ring_slots(c, 8192, 128) == 1152
    assert capacity_bytes(c, 8192, 128) == st.nbytes - st.pos.numel() * 4
    with pytest.raises(ValueError, match="ring reserve"):
        st.check(129)
    with pytest.raises(ValueError):
        attention.check_geometry(Config(**{**config().__dict__, "global_head_dim": 256}))
    with pytest.raises(ValueError, match="int32"):
        attention.check_geometry(Config(**{**config().__dict__, "max_position": 1 << 21}))


@pytest.mark.parametrize("window,noncausal", [(16, False), (16, True), (0, True), (0, False)])
@pytest.mark.parametrize("total", [300, 1026])
def test_companion_geometry_ring_attention_matches_float64_oracle(window, noncausal, total):
    """Head width 128, query group 4, softmax scale, unbounded window and a bidirectional proposal block."""
    torch.manual_seed(7)
    block, hk, g, d = 5, 8, 4, 128
    ring = 384 if window else -(-total // 64) * 64
    keys = (torch.randn(ring, hk, d) * 2).to(torch.bfloat16).to(DEV)
    values = (torch.randn(ring, hk, d) * 2).to(torch.bfloat16).to(DEV)
    q = (torch.randn(block, hk * g, d) * 2).to(torch.bfloat16).to(DEV)
    start = total - block
    pos = torch.tensor([start], dtype=torch.int32, device=DEV)
    bs = torch.full((block,), start, dtype=torch.int32, device=DEV)
    be = bs + block
    scale = d ** -0.5
    if window:
        slots, chunk = sliding_slots(window, block, g, block if noncausal else 0), attention.SLIDING_CHUNK
    else:
        chunk = 1024 if noncausal else attention.SLIDING_CHUNK
        slots = (total - 1) // chunk + 1
    out = attention.sliding(q, keys, values, pos, bs, be, window=window, slots=slots, blocks=noncausal,
                            scale=scale, chunk=chunk)
    # Wrong explicit bounds deliberately remain readable: the uniform variant
    # must derive every active/tail row from pos and block, ignoring these values.
    ignored = torch.full_like(bs, -1)
    uniform = attention.sliding(q, keys, values, pos, ignored, ignored, window=window, slots=slots,
                                blocks=noncausal, scale=scale, chunk=chunk, uniform_block=True)
    assert torch.equal(uniform, out)
    index = torch.arange(total, device=DEV) % ring
    k, v = keys[index], values[index]
    qpos = start + torch.arange(block, device=DEV)
    kpos = torch.arange(total, device=DEV)
    mask = kpos[None, :] <= qpos[:, None]
    if noncausal:
        mask |= kpos[None, :] >= start
    if window:
        mask &= kpos[None, :] >= qpos[:, None] - window + 1
    kk = k.double().repeat_interleave(g, 1).transpose(0, 1)
    vv = v.double().repeat_interleave(g, 1).transpose(0, 1)
    scores = torch.einsum("thd,hkd->htk", q.double(), kk) * scale
    expected = torch.einsum("htk,hkd->thd", torch.softmax(scores.masked_fill(~mask[None], float("-inf")), -1), vv)
    within_bf16(out, expected, v)

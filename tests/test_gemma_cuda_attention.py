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


@pytest.mark.parametrize("window,noncausal", [(16, False), (0, True), (0, False)])
def test_companion_geometry_ring_attention_matches_float64_oracle(window, noncausal):
    """Head width 128, query group 4, softmax scale, unbounded window and a bidirectional proposal block."""
    torch.manual_seed(7)
    total, block, hk, g, d = 300, 5, 8, 4, 128
    ring = 384 if window else 512
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


def _query_rows_storage(q, chunks, held):
    """Publish each allocation before attempting the next, including failure."""
    rows, heads, dim = q.shape
    held[0] = torch.empty_like(q)
    held[1] = torch.empty((rows, heads, chunks, dim), dtype=torch.float32, device=DEV)
    held[2] = torch.empty((rows, heads, chunks), dtype=torch.float32, device=DEV)
    held[3] = torch.empty((rows, heads, chunks), dtype=torch.float32, device=DEV)



def _query_rows_inputs(c, total, held):
    """Same seed19 draws/roundings as Stream, with incremental GPU ownership."""
    held['generator'] = torch.Generator(device="cpu")
    held['generator'].manual_seed(19)
    generator = held['generator']
    # Keep the original CPU RNG stream, without unused sliding GPU tensors.
    for shape in ((total, c.heads * c.head_dim), (total, c.kv_heads * c.head_dim),
                  (total, c.kv_heads * c.head_dim)):
        held['discard'] = torch.randn(*shape, generator=generator)
    for name, shape in (("gq", (total, c.heads * c.global_head_dim)),
                        ("gk", (total, c.global_kv_heads * c.global_head_dim))):
        stages = held[name]
        stages[0] = torch.randn(*shape, generator=generator)
        stages[1] = stages[0] * 3.0
        stages[2] = stages[1].to(torch.bfloat16)
        stages[3] = stages[2].to(DEV)
    for _ in range(2):  # original sliding qw and kw draws
        held['discard'] = torch.randn(c.head_dim, generator=generator)
    for name in ("gqw", "gkw"):
        stages = held[name]
        stages[0] = torch.randn(c.global_head_dim, generator=generator)
        stages[1] = stages[0] * 3.0
        stages[2] = stages[1].to(torch.bfloat16)
        stages[3] = stages[2].to(DEV)
        stages[4] = stages[3] * 0.3
        stages[5] = stages[4] + 1


def _query_rows_frequency(c, held):
    """Original inverse() GPU operation order, with each temporary owned."""
    dim = c.global_head_dim
    rotated = int(dim * c.global_rotary_fraction)
    held['indices'] = torch.arange(0, dim, 2, dtype=torch.float32, device=DEV)
    held['negative'] = -held['indices']
    held['exponent'] = held['negative'] / dim
    held['value'] = torch.pow(c.global_rope_theta, held['exponent'])
    held['value'][rotated // 2:] = 0

def _launch_query_rows(q, raw, inv, weight, freq, pos, chunks, query_rows, held, *, sequential=False):
    """Private specialization only; production full() retains its default grid."""
    rows, heads, dim = q.shape
    kv_heads = raw.shape[1]
    out, po, pm, pl = held
    attention._global[((rows * (heads // kv_heads) + query_rows - 1) // query_rows,
                       kv_heads, 1 if sequential else chunks)](
        q, raw, inv, weight, freq, raw, raw, pos, pos, pos, po, pm, pl, out, rows, chunks,
        H=heads, HK=kv_heads, D=dim, G=heads // kv_heads, ROT=attention.ROTATED,
        UW=(dim - 2 * attention.ROTATED) // 3, CH=attention.GLOBAL_CHUNK,
        BK=attention.GLOBAL_TILE, SEQUENTIAL=sequential, ROUND=False, PREPARED=False,
        TRIG=False, KEYS_ONLY=False, QUERY_ROWS=query_rows, **attention.LAUNCH)
    if not sequential:
        attention._merge[(rows, heads, dim // attention.MERGE_COLUMNS)](
            po, pm, pl, out, pos, chunks, H=heads, D=dim, DS=attention.MERGE_COLUMNS,
            CH=attention.GLOBAL_CHUNK, GLOBAL=True, num_warps=1, enable_fp_fusion=False)
    return out


def _same_words(left, right):
    assert left.shape == right.shape and left.dtype == right.dtype
    left_contiguous = left.contiguous()
    right_contiguous = right.contiguous()
    left_bytes = left_contiguous.view(torch.uint8)
    right_bytes = right_contiguous.view(torch.uint8)
    assert torch.equal(left_bytes, right_bytes)


def _capture_query_rows(graph, body, held):
    """No context/finally CUDA teardown: isolated failure transport owns frames.

    The supervised test boundary retains a failed traceback and every published
    owner before retiring the process. Never end capture or restore streams on
    an unknown CUDA failure. Ordinary successful execution restores the stream.
    """
    held['graph'] = graph
    held['previous'] = torch.cuda.current_stream()
    held['stream'] = torch.cuda.Stream()
    held['stream'].wait_stream(held['previous'])
    torch.cuda.set_stream(held['stream'])
    graph.capture_begin(capture_error_mode='global')
    held['outputs'] = body()
    graph.capture_end()
    torch.cuda.set_stream(held['previous'])
    held['previous'].wait_stream(held['stream'])


def _require_query_rows_target():
    if torch.cuda.get_device_capability() != (12, 1):
        pytest.skip("private query-row specialization is qualified only on SM121")


@pytest.mark.parametrize("rows", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("position", [0, 511, 512])
def test_private_global_query_rows_match_defined_partials_and_serial(rows, position):
    _require_query_rows_target()
    c = config()
    total = position + rows
    inputs = {'generator': None, 'discard': None, 'gq': [None] * 4, 'gk': [None] * 4,
              'gqw': [None] * 6, 'gkw': [None] * 6}
    _query_rows_inputs(c, total, inputs)
    state = State(c, DEV, 4096, rows=128)
    frequency = {'indices': None, 'negative': None, 'exponent': None, 'value': None}
    _query_rows_frequency(c, frequency)
    freq = frequency['value']
    all_q = attention.prep_full(inputs['gq'][3], inputs['gk'][3], inputs['gqw'][5], freq, state.pos,
                                state.raw[0], state.inv[0], eps=EPS, round_normalized=False)
    q = all_q[position:position + rows]
    state.pos.fill_(position)
    chunks = state.context // attention.GLOBAL_CHUNK
    held = {16: [None] * 4, 8: [None] * 4}
    for width in (16, 8):
        _query_rows_storage(q, chunks, held[width])
    expected = attention.full(q, state.raw[0], state.inv[0], inputs['gkw'][5], freq, state.pos,
                              chunks=chunks, round_normalized=False, sequential=False)
    for width in (16, 8):
        actual = _launch_query_rows(q, state.raw[0], state.inv[0], inputs['gkw'][5], freq,
                                    state.pos, chunks, width, held[width])
        _same_words(actual, expected)
    # Only chunks consumed by this row's merge are necessarily written in both
    # layouts. At P511/M2, row0/chunk1 is unwritten under QT8, not a zero oracle.
    for row in range(rows):
        reachable = (position + row) // attention.GLOBAL_CHUNK + 1
        for baseline, candidate in zip(held[16][1:], held[8][1:]):
            _same_words(baseline[row, :, :reachable], candidate[row, :, :reachable])
    for width in (16, 8):
        _same_words(_launch_query_rows(q, state.raw[0], state.inv[0], inputs['gkw'][5], freq,
                                      state.pos, chunks, width, held[width], sequential=True), expected)
    serial = [None] * rows
    for row in range(rows):
        state.pos.fill_(position + row)
        serial[row] = attention.full(q[row:row + 1], state.raw[0], state.inv[0], inputs['gkw'][5], freq,
                                     state.pos, chunks=chunks, round_normalized=False, sequential=False)
    serial_join = torch.cat(serial)
    _same_words(serial_join, expected)


@pytest.mark.parametrize("rows", [1, 2, 3, 4, 5])
def test_private_global_query_rows_graph_uses_live_position_and_restored_inputs(rows):
    _require_query_rows_target()
    c = config()
    state = State(c, DEV, 4096, rows=128)
    # A visible key immediately after the boundary dominates column64, outside
    # both rotated halves. This makes stale position observably wrong.
    state.raw.zero_()
    state.inv.fill_(1)
    weight = torch.ones(512, dtype=torch.bfloat16, device=DEV)
    q = torch.zeros((rows, c.heads, 512), dtype=torch.bfloat16, device=DEV)
    q[:, :, 64] = 16
    state.raw[0, 513, :, 64] = 16
    frequency = {'indices': None, 'negative': None, 'exponent': None, 'value': None}
    _query_rows_frequency(c, frequency)
    freq = frequency['value']
    chunks = state.context // attention.GLOBAL_CHUNK
    held = {16: [None] * 4, 8: [None] * 4}
    for width in (16, 8):
        _query_rows_storage(q, chunks, held[width])
    graphs = {16: None, 8: None}
    captures = {16: {'graph': None, 'previous': None, 'stream': None, 'outputs': None},
                8: {'graph': None, 'previous': None, 'stream': None, 'outputs': None}}
    for width in (16, 8):
        graphs[width] = torch.cuda.CUDAGraph()
    state.pos.fill_(512)
    for width in (16, 8):
        _launch_query_rows(q, state.raw[0], state.inv[0], weight, freq, state.pos, chunks, width, held[width])
    torch.cuda.synchronize()
    for width in (16, 8):
        _capture_query_rows(graphs[width],
                            lambda w=width: _launch_query_rows(q, state.raw[0], state.inv[0], weight, freq,
                                                               state.pos, chunks, w, held[w]), captures[width])
    outputs = [None] * 4
    for index, position in enumerate((512, 513, 510, 512)):
        state.pos.fill_(position)
        expected = attention.full(q, state.raw[0], state.inv[0], weight, freq, state.pos,
                                  chunks=chunks, round_normalized=False, sequential=False)
        for width in (16, 8):
            graphs[width].replay()
            _same_words(held[width][0], expected)
        outputs[index] = expected
    assert not torch.equal(outputs[0][0].view(torch.uint8), outputs[1][0].view(torch.uint8))
    _same_words(outputs[0], outputs[-1])
    original = state.raw[0, 513].clone()
    state.raw[0, 513].zero_()
    state.pos.fill_(513)
    changed = attention.full(q, state.raw[0], state.inv[0], weight, freq, state.pos,
                             chunks=chunks, round_normalized=False, sequential=False)
    assert not torch.equal(changed.view(torch.uint8), outputs[1].view(torch.uint8))
    for width in (16, 8):
        graphs[width].replay()
        _same_words(held[width][0], changed)
    state.raw[0, 513].copy_(original)
    for width in (16, 8):
        graphs[width].replay()
        _same_words(held[width][0], outputs[1])

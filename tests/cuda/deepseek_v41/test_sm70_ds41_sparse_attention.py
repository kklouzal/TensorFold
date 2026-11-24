"""Fused sparse attention (ds41_sparse_attention, patch 181 at TP8) vs patch 87's gather + bmm path and an FP64
oracle, on packed pool pages, slot rings and multi-request steps; run as a script for the single-GPU timings.

    pytest -q test_sm70_ds41_sparse_attention.py
    python test_sm70_ds41_sparse_attention.py OUT.json      # timings (8K / 16K prompt chunks, decode M=1..32)
"""

import json
import sys
from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

H, BLOCK = 8, 64
SOURCES = {2: 2, 8: 2, 14: 2, 20: 1}       # the lane's KV sources and ratios (config compress_ratios)


def build(requests, *, source=2, compressed=True, qscale=2.0, seed=0, dev="cuda"):
    """A step of ``requests`` [(start, tokens)] on the packed pool: (args, kwargs) of mu_packed.attend.

    Earlier positions of each request are in its pages (compressed rows from random latents) and its slot ring;
    top-512 positions are random distinct live rows per token (all live rows below 512, -1 padded)."""
    from tensorfold.families.deepseek_v4.cuda.kernels import mu_packed as pk

    g = torch.Generator(device=dev).manual_seed(seed)
    layout = pk.PackedLayout(SOURCES, BLOCK)
    ratio = SOURCES[source]
    R = len(requests)
    pages_per = [-(-(s + t) // BLOCK) for s, t in requests]
    width = max(pages_per)
    order = torch.randperm(sum(pages_per), generator=g, device=dev).int()
    table = torch.full((R, width), -1, dtype=torch.int32, device=dev)
    at = 0
    for r, n in enumerate(pages_per):
        table[r, :n] = order[at:at + n]
        at += n
    cache = torch.zeros(sum(pages_per), layout.page_bytes, dtype=torch.uint8, device=dev)
    slots = R + 2
    ring = (torch.randn(slots + 1, pk.RING, 512, generator=g, device=dev) * 0.3).half()
    for r, (s, t) in enumerate(requests):          # every earlier position's compressed row in the pages
        p = torch.arange(s + t, device=dev)
        latent = (torch.randn(s + t, 512, generator=g, device=dev) * 0.3).half()
        pk.insert_rows(cache, latent, p, torch.ones_like(p, dtype=torch.bool), torch.full_like(p, r, dtype=torch.int32),
                       table, layout=layout, layer=source, compressed=True)
    T = sum(t for _, t in requests)
    positions = torch.cat([torch.arange(s, s + t, device=dev) for s, t in requests])
    token_req = torch.cat([torch.full((t,), r, dtype=torch.int32, device=dev) for r, (_, t) in enumerate(requests)])
    qsl = torch.tensor([0] + list(torch.tensor([t for _, t in requests]).cumsum(0)), dtype=torch.int32, device=dev)
    req_slot = torch.tensor([r + 1 for r in range(R)], dtype=torch.int32, device=dev)
    valid = torch.ones(T, dtype=torch.bool, device=dev)
    if T > 3:
        valid[T // 3] = False                       # an invalid row (padding) in the middle
    step = SimpleNamespace(valid=valid, token_req=token_req, qsl=qsl, spec=None)
    query = (torch.randn(T, H, 512, generator=g, device=dev) * qscale).half()
    window = (torch.randn(T, 512, generator=g, device=dev) * 0.3).half()
    sink = torch.randn(H, generator=g, device=dev)
    kw = {}
    if compressed:
        n_c = (positions + 1) // ratio
        topk = torch.empty(T, 512, dtype=torch.int32, device=dev)
        for a in range(0, T, 512):
            b = min(a + 512, T)
            top = int(n_c[a:b].max())
            score = torch.rand(b - a, max(top, 512), generator=g, device=dev)
            col = torch.arange(score.shape[1], device=dev)
            score = torch.where(col[None] < n_c[a:b, None], score, -1.0)
            v, i = score.topk(512, dim=1)
            topk[a:b] = torch.where(v >= 0, i, -1).int()
        kw = dict(cache=cache, block_table=table, layout=layout, source=source, topk=topk)
    return (query, positions, step, req_slot, ring, window, sink), kw


def reference(args, kw):
    from tensorfold.families.deepseek_v4.cuda.kernels import mu_packed as pk
    return pk.attend(*args, **kw)


def fused(args, kw, **extra):
    from tensorfold.cuda.sm70 import ds41_sparse_attention as sa
    return sa.attend(*args, **kw, **extra)


def encode_rotor_rows(rows):
    from tensorfold.cuda.sm70 import rotorquant_q4_native as codec

    matrix = rows.reshape(-1, 1, 512)
    n = matrix.shape[0]
    codes = torch.empty(n, 1, 256, dtype=torch.uint8, device=rows.device)
    scales = torch.empty(n, 1, 4, dtype=torch.float32, device=rows.device)
    codec.encode(matrix, codes, scales, codec=11)
    decoded = torch.empty_like(matrix)
    codec.decode(codes, scales, decoded, codec=11)
    wire = torch.cat((codes.view(n, 256), scales.view(torch.uint8).view(n, 16)), dim=1)
    return wire.reshape(*rows.shape[:-1], 272), decoded.view_as(rows)


def test_fused_rotor_window_matches_explicit_decode_on_replay():
    args, kw = build([(129, 4), (0, 3)], compressed=False, seed=52)
    query, positions, step, req_slot, ring, window, sink = args
    packed_ring, _ = encode_rotor_rows(ring)
    packed_window, _ = encode_rotor_rows(window)
    operands = (query, positions, step, req_slot, packed_ring, packed_window, sink)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        fused(operands, kw)
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        actual = fused(operands, kw)

    for scale in (1, -2):
        ring_bytes, ring_half = encode_rotor_rows(ring * scale)
        window_bytes, window_half = encode_rotor_rows(window * scale)
        packed_ring.copy_(ring_bytes)
        packed_window.copy_(window_bytes)
        want = fused((query, positions, step, req_slot, ring_half, window_half, sink), kw)
        graph.replay()
        assert torch.equal(actual.view(torch.int16), want.view(torch.int16))

    with pytest.raises(RuntimeError, match="values must match the window-ring format"):
        fused((query, positions, step, req_slot, packed_ring, window, sink), kw)


@pytest.mark.parametrize("source", [2, 20])
def test_fused_rotor_paged_latent_matches_single_selected_key(source):
    from tensorfold.families.deepseek_v4.cuda.kernels.mu_packed import PackedLayout

    args, legacy = build([(62, 4), (126, 4), (0, 2)], source=source, seed=81)
    query, positions, step, req_slot, ring, window, sink = args
    query.zero_()
    sink.zero_()
    layout = PackedLayout(SOURCES, BLOCK, rotorquant=True)
    ratio, per, offset, _ = layout.regions[source]
    table = legacy["block_table"].clone()
    table[1, 1] = -1
    pages = legacy["cache"].shape[0]
    cache = torch.full((pages, layout.page_bytes), 255, dtype=torch.uint8, device=query.device)
    rng = torch.Generator(device=query.device).manual_seed(34)
    latents = (0.7 + torch.rand((pages, per, 512), generator=rng, device=query.device)).half()
    wire, decoded = encode_rotor_rows(latents)
    cache[:, offset:offset + per * 272].copy_(wire.flatten(1))
    # A zero code and zero scale represent a zero row. Its sign has no impact on the one-term PV result.
    packed_ring = torch.zeros((*ring.shape[:-1], 272), dtype=torch.uint8, device=query.device)
    packed_window = torch.zeros((len(window), 272), dtype=torch.uint8, device=query.device)
    topk = torch.full_like(legacy["topk"], -1)
    newest = (positions + 1) // ratio - 1
    topk[:, 0] = newest.int()
    topk[:, 1] = newest.int() + 1  # a causally future key must be masked
    kw = dict(cache=cache, block_table=table, layout=layout, source=source, topk=topk)
    actual = fused((query, positions, step, req_slot, packed_ring, packed_window, sink), kw)

    expected = torch.zeros_like(actual)
    requests = step.token_req.tolist()
    live = step.valid.tolist()
    page_ids = table.tolist()
    for row, (pos, req, valid, index) in enumerate(zip(positions.tolist(), requests, live, newest.tolist())):
        if not valid or index < 0:
            continue
        page = page_ids[req][index // per]
        if page >= 0:
            probability = torch.tensor(1 / (min(pos + 1, 128) + 2), device=query.device).half()
            expected[row] = (decoded[page, index % per].float() * probability.float()).half()
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))

    with pytest.raises(ValueError, match="compressed row format must match"):
        fused((query, positions, step, req_slot, packed_ring, packed_window, sink), legacy)
    with pytest.raises(ValueError, match="compressed row format must match"):
        fused(args, kw)


@pytest.mark.parametrize("start", [0, 193])
@pytest.mark.parametrize("source", [None, 2, 20])
def test_h64_prefill_decoded_current_matches_packed_reader(start, source, monkeypatch):
    from tensorfold.cuda.sm70 import ds41_sparse_attention as sa
    from tensorfold.families.deepseek_v4.cuda.kernels.mu_packed import PackedLayout
    from tensorfold.families.deepseek_v4.cuda.target import Target

    tokens = 1024
    args, kw = build([(start, tokens)], source=source or 2, compressed=source is not None, seed=93)
    query, positions, step, req_slot, ring, window, sink = args
    window[:, 64:128] *= 5
    window[::3, 128:192] = -0.0
    packed_ring, _ = encode_rotor_rows(ring)
    packed_window, _ = encode_rotor_rows(window)
    operands = (query, positions, step, req_slot, packed_ring, packed_window, sink)
    step.positions, step.req_slot = positions, req_slot
    step.image, step.num_reqs = None, 1
    step.prefill = [(0, 0, tokens, start + tokens - 1)]
    if source is not None:
        layout = PackedLayout(SOURCES, BLOCK, rotorquant=True)
        _ratio, per_page, offset, _ = layout.regions[source]
        pages = kw["cache"].shape[0]
        generator = torch.Generator(device=query.device).manual_seed(27)
        latents = (torch.randn((pages, per_page, 512), device=query.device, generator=generator) * 0.45).half()
        packed_latents, _ = encode_rotor_rows(latents)
        cache = torch.zeros((pages, layout.page_bytes), device=query.device, dtype=torch.uint8)
        cache[:, offset:offset + per_page * 272].copy_(packed_latents.flatten(1))
        kw.update(cache=cache, layout=layout)

    want = fused(operands, kw)
    attend = sa.attend
    selected = []

    def record(*args, **kwargs):
        selected.append(kwargs["decode_window"])
        return attend(*args, **kwargs)

    monkeypatch.setattr(sa, "attend", record)
    target = Target.__new__(Target)
    kwargs = dict(kw)
    if source is not None:
        kwargs["layer"] = kwargs.pop("source")
    got = target._attend(query, step, packed_ring, packed_window, sink, **kwargs)
    expand_in_serving = source is not None and start == 0
    assert selected == [expand_in_serving]
    assert torch.equal(got.view(torch.int16), want.view(torch.int16))
    assert not torch.count_nonzero(got[tokens // 3])   # inactive query inside the current chunk
    if not expand_in_serving:
        expanded = attend(*operands, **kw, decode_window=True)
        assert torch.equal(expanded.view(torch.int16), want.view(torch.int16))

    if start and source is None:
        # The last old age is visible at local row 126 and absent at 127.
        # A changed historical row must affect exactly that side of the seam.
        changed = packed_ring.clone()
        changed[1, (start - 1) % 128].copy_(packed_window[0])
        changed_args = (query, positions, step, req_slot, changed, packed_window, sink)
        reference = attend(*changed_args)
        candidate = attend(*changed_args, decode_window=True)
        assert torch.equal(candidate.view(torch.int16), reference.view(torch.int16))
        assert not torch.equal(candidate[126].view(torch.int16), got[126].view(torch.int16))
        assert torch.equal(candidate[127:].view(torch.int16), got[127:].view(torch.int16))
        with pytest.raises(RuntimeError, match="single memory location"):
            attend(*operands, out=query, decode_window=True)
        with pytest.raises(RuntimeError, match="single-request H64 step"):
            attend(*args, decode_window=True)


@pytest.mark.parametrize("requests,compressed", [([(127, 1)], True),
    ([(127, 1), (128, 1), (16384, 1), (0, 1)], True),
    ([(127, 1), (128, 1), (16384, 1), (0, 1)], False)])
def test_decode_window_ring_store_matches_post_attention_commit(requests, compressed):
    from tensorfold.families.deepseek_v4.cuda.kernels import mu_packed as pk
    from tensorfold.families.deepseek_v4.cuda.kernels.quant_rope import rope_table
    from tensorfold.families.deepseek_v4.cuda.kernels.rope_qdq import rope_qdq

    args, kw = build(requests, compressed=compressed, seed=120)
    query, positions, step, req_slot, initial, value, sink = args
    rows = len(requests)
    live_rows = rows
    if rows > 1:
        req_slot[-1] = initial.shape[0] - 1
        live_rows -= 1  # graph padding keeps the scratch slot at position zero
    table = rope_table(compressed=compressed)
    old_ring, new_ring = initial.clone(), initial.clone()
    attend = fused if rows > 1 and not compressed else reference

    def forward(ring, early):
        write = (ring, req_slot, step.valid) if early else None
        window = rope_qdq(value, positions, table, decode_ring=write)
        output = attend((query, positions, step, req_slot, ring, window, sink), kw)
        if not early:
            pk.insert_ring(ring, window, positions, step.valid, step.token_req, step.qsl, req_slot)
        return window, output

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        forward(new_ring, True)
        new_ring.copy_(initial)
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        actual = forward(new_ring, True)

    for _ in range(3):
        expected = forward(old_ring, False)
        graph.replay()
        assert all(torch.equal(x.view(torch.int16), y.view(torch.int16)) for x, y in zip(actual, expected, strict=True))
        assert torch.equal(old_ring.view(torch.int16), new_ring.view(torch.int16))
        assert torch.equal(new_ring[-1], initial[-1])
        positions[:live_rows].add_(1)
        value.neg_()


@pytest.mark.parametrize("rows,compressed,eps",
    [(1, False, 1e-20), (8, True, 1e-20), (32, True, 1e-20), (64, False, 1e-6)])
def test_decode_window_rms_matches_separate_launch(rows, compressed, eps):
    from tensorfold.families.deepseek_v4.cuda.kernels.mu_packed import RING
    from tensorfold.families.deepseek_v4.cuda.kernels.norm import rms_norm
    from tensorfold.families.deepseek_v4.cuda.kernels.quant_rope import rope_table
    from tensorfold.families.deepseek_v4.cuda.kernels.rope_qdq import rope_qdq

    device = "cuda"
    rng = torch.Generator(device=device).manual_seed(407)
    scale = torch.logspace(-3, 1, 512, device=device)
    value = torch.empty((rows, 512), device=device, dtype=torch.float16)
    weight = torch.rand(512, device=device, dtype=torch.float32, generator=rng).add_(0.5)
    positions = torch.tensor([127, 128, 4095, 16383, 16384], device=device).repeat((rows + 4) // 5)[:rows]
    slots = torch.arange(rows, device=device, dtype=torch.int32)
    valid = torch.ones(rows, device=device, dtype=torch.bool)
    if rows > 1:
        valid[-2] = False
        slots[-1] = rows  # scratch
        positions[-1] = 0
    live_rows = rows if rows == 1 else rows - 1
    table = rope_table(compressed=compressed)
    initial = torch.randn((rows + 1, RING, 512), device=device, generator=rng).half()
    old_ring, new_ring = initial.clone(), initial.clone()

    def combined():
        return rope_qdq(value, positions, table, decode_ring=(new_ring, slots, valid), decode_norm=(weight, eps))

    value.zero_()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        combined()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        captured = combined()
    new_ring.copy_(initial)

    for replay in range(5):
        if replay == 1:
            # Exercise unequal rotary partners, including signed zero and small half values.
            value[:, -64::2] = 2.0**-14
            value[:, -63::2] = -0.0
            value[:, -1] = 1.0
        elif replay > 1:
            value.copy_(torch.randn(value.shape, device=device, generator=rng) * scale)
        expected = rope_qdq(rms_norm(value, weight, eps), positions, table, decode_ring=(old_ring, slots, valid))
        graph.replay()
        assert torch.equal(captured.view(torch.int16), expected.view(torch.int16))
        assert torch.equal(new_ring.view(torch.int16), old_ring.view(torch.int16))
        positions[:live_rows].add_(1)


def oracle(args, kw, rows):
    """FP64 attention of token ``rows`` over the reference's gathered FP16 keys (exact decodes of the pages)."""
    from tensorfold.families.deepseek_v4.cuda.kernels import mu_packed as pk
    query, positions, step, req_slot, ring, window, sink = args
    width = pk.WINDOW + (512 if kw else 0)
    out = []
    for t in rows.tolist():
        keys = torch.empty(1, width, 512, dtype=torch.float16, device=query.device)
        unused = torch.empty(1, width, dtype=torch.bool, device=query.device)
        pk.gather_attention(keys, unused, t, t + 1, positions, step.valid, step.token_req, step.qsl, req_slot,
                            ring, window, cache=kw.get("cache"), block_table=kw.get("block_table"),
                            layout=kw.get("layout"), layer=kw.get("source"), topk=kw.get("topk"))
        k = keys[0].double()
        s = (query[t].double() @ k.T) * 512 ** -0.5
        s = torch.where(unused[0][None], float("-inf"), s)
        z = sink.double()[:, None]
        m = torch.maximum(s.max(1, keepdim=True).values, z)
        e = (s - m).exp()
        p = e / (e.sum(1, keepdim=True) + (z - m).exp())
        out.append(p @ k)
    return torch.stack(out)


def compare(args, kw, ref, new, samples=96):
    rows = torch.linspace(0, ref.shape[0] - 1, min(samples, ref.shape[0]), device=ref.device).long().unique()
    exact = oracle(args, kw, rows)
    def err(x):
        d = x[rows].double() - exact
        return float(d.abs().max()), float(d.norm() / exact.norm())
    same = float((ref.view(torch.int16) == new.view(torch.int16)).float().mean())
    return {"equal_fp16_words": same, "max_abs_ref_vs_new": float((ref.float() - new.float()).abs().max()),
            "ref_vs_oracle": err(ref), "new_vs_oracle": err(new)}


CASES = [
    ("prefill 1 req from 0, ratio 2", [(0, 1536)], 2, True),
    ("prefill chunk after 6K (ring + pages), ratio 1", [(6144, 1024)], 20, True),
    ("mixed: 2 prefill + 3 decode rows, ratio 2", [(0, 700), (3000, 300), (5000, 1), (77, 1), (600, 1)], 8, True),
    ("window-only layer, chunk after 200", [(200, 900), (0, 3)], 0, False),
]


@pytest.mark.parametrize("name,requests,source,compressed", CASES, ids=[c[0] for c in CASES])
@pytest.mark.parametrize("qscale", [0.5, 4.0])
def test_fused_matches_patch87_and_oracle(name, requests, source, compressed, qscale):
    args, kw = build(requests, source=source if compressed else 2, compressed=compressed, qscale=qscale)
    ref, new = reference(args, kw), fused(args, kw)
    torch.cuda.synchronize()
    assert torch.isfinite(new).all()
    stats = compare(args, kw, ref, new)
    print(name, qscale, stats)
    # FP32 accumulation in either order: the fused error vs FP64 stays within the reference's own (plus one
    # FP16 ulp of the output scale) and outputs agree to a few FP16 ulps.
    ref_abs, ref_rel = stats["ref_vs_oracle"]
    new_abs, new_rel = stats["new_vs_oracle"]
    scale = float(ref.float().abs().max())
    assert new_rel <= ref_rel * 1.5 + 1e-4, stats
    assert new_abs <= ref_abs + scale * 2 ** -10, stats
    assert stats["max_abs_ref_vs_new"] <= scale * 2 ** -8, stats


def test_invalid_rows_and_rejections():
    args, kw = build([(0, 64)], source=2)
    new = fused(args, kw)
    ref = reference(args, kw)
    torch.cuda.synchronize()
    dead = ~args[2].valid
    # invalid rows: no live key, the sink alone -> zero output, as in the reference
    assert torch.equal(new[dead], ref[dead]) and float(new[dead].abs().max()) == 0.0
    with pytest.raises(RuntimeError):
        fused((args[0][:, :4].contiguous(),) + args[1:], kw)       # 4 heads: refused
    with pytest.raises(RuntimeError):
        bad = dict(kw, topk=kw["topk"][:, :256].contiguous())
        fused(args, bad)


def _time(fn, reps):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        b.synchronize()
        times.append(a.elapsed_time(b))
    times.sort()
    return {"median_ms": times[len(times) // 2], "min_ms": times[0], "max_ms": times[-1], "reps": reps}


def bench(out_path):
    results = {"device": torch.cuda.get_device_name(), "cases": []}
    shapes = [("chunk 8192 from 0", [(0, 8192)]), ("chunk 8192 from 8192 (16K prompt)", [(8192, 8192)])]
    for label, requests in shapes:
        for source, compressed in ((2, True), (20, True), (2, False)):
            args, kw = build(requests, source=source, compressed=compressed, qscale=2.0)
            ref, new = reference(args, kw), fused(args, kw)
            torch.cuda.synchronize()
            row = {"shape": label, "source": source if compressed else None, "ratio": SOURCES[source] if compressed else 0,
                   "numerics": compare(args, kw, ref, new, samples=48),
                   "patch87": _time(lambda: reference(args, kw), 10),
                   "fused": _time(lambda: fused(args, kw), 10)}
            row["speedup"] = row["patch87"]["median_ms"] / row["fused"]["median_ms"]
            print(json.dumps(row), flush=True)
            results["cases"].append(row)
            del args, kw, ref, new
            torch.cuda.empty_cache()
    for m in (1, 2, 4, 8, 16, 24, 32):
        requests = [(4000 + 131 * r, 1) for r in range(m)]
        args, kw = build(requests, source=2, qscale=2.0)
        out = torch.empty_like(args[0])
        row = {"shape": f"decode M={m}", "source": 2, "patch87": _time(lambda: reference(args, kw), 50),
               "fused": _time(lambda: fused(args, kw, out=out), 50)}
        print(json.dumps(row), flush=True)
        results["cases"].append(row)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=1)


def _graph_time(fn, reps=200):
    """Median per-call time of ``fn`` replayed from a CUDA graph of 20 back-to-back calls (decode-graph conditions)."""
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        with torch.cuda.graph(graph, stream=stream):
            for _ in range(20):
                fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(reps // 20):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        graph.replay()
        b.record()
        b.synchronize()
        times.append(a.elapsed_time(b) / 20)
    times.sort()
    return times[len(times) // 2]


def bench_decode_graphs(out_path):
    """Decode-size steps (M = 1..64 one-token requests) of every layer kind, timed inside CUDA graphs."""
    rows = []
    for source, compressed in ((2, True), (20, True), (2, False)):
        for m in (1, 2, 4, 8, 16, 24, 32, 48, 64):
            requests = [(4000 + 131 * r, 1) for r in range(m)]
            args, kw = build(requests, source=source, compressed=compressed, qscale=2.0)
            out = torch.empty_like(args[0])
            row = {"m": m, "ratio": SOURCES[source] if compressed else 0,
                   "patch87_ms": _graph_time(lambda: reference(args, kw)),
                   "fused_ms": _graph_time(lambda: fused(args, kw, out=out))}
            print(json.dumps(row), flush=True)
            rows.append(row)
    with open(out_path, "w") as f:
        json.dump({"device": torch.cuda.get_device_name(), "graph_rows": rows}, f, indent=1)


if __name__ == "__main__":
    (bench_decode_graphs if sys.argv[1:2] == ["--graphs"] else bench)(sys.argv[-1])

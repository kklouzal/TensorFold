"""Gemma 4 SM70 native kernels (cuda/sm70/gemma4.cu) against FP64 references of their declared contracts.

W4A16 GEMV/GEGLU/expert kernels must equal exact E2M1 x E4M3 x row-multiplier weights up to FP32 summation;
dequantized FP16 weights are exact; routing matches softmax/top-k/renormalize/per-expert scale; the WMMA
split-key attention matches masked softmax attention (FP16 operands) for causal, sliding-ring and split cases.
"""
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0),
                                reason="needs an SM70 GPU")
E2M1 = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6], dtype=torch.float64)

def e4m3(b):
    b = b.long(); s = torch.where(b & 128 > 0, -1.0, 1.0).double(); e = (b >> 3) & 15; m = b & 7
    v = torch.where(e == 0, m.double() / 8 * 2.0 ** -6, (1 + m.double() / 8) * 2.0 ** (e.double() - 7))
    return s * v

def rand_packed(n, k):
    codes = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8)
    scales = torch.randint(0, 0x7f, (n, k // 16), dtype=torch.uint8)  # finite, includes subnormals
    scales[(scales & 0x7f) == 0x7f] = 0x30
    rowmul = torch.rand(n) * 1e-3 + 1e-4
    return codes, scales, rowmul

def dequant(codes, scales, rowmul):
    n = codes.shape[0]
    c = torch.empty((n, codes.shape[1] * 2), dtype=torch.long)
    c[:, 0::2] = (codes & 15).long(); c[:, 1::2] = (codes >> 4).long()
    return E2M1[c] * torch.repeat_interleave(e4m3(scales), 16, 1) * rowmul.double()[:, None]

def check(name, got, ref, tol):
    got = got.double().cpu(); ref = ref.double().cpu()
    err = ((got - ref).abs().max() / ref.abs().max().clamp_min(1e-30)).item()
    assert err < tol, f"{name}: relative error {err:.3g} >= {tol}"

def tab(*tensors):
    """Slot address table (int64 device) of per-slot cache tensors."""
    return torch.tensor([t.data_ptr() for t in tensors], dtype=torch.int64, device="cuda")

def segs(segments, H):
    """Native segment table [6, width] for segments [(slot, pos, rows)] laid out consecutively; tiles count."""
    tiles, row0 = [], 0
    for g, (_, _, rows) in enumerate(segments):
        tiles += [(g, r) for r in range(0, rows * H, 16)]
    width = max(len(segments), len(tiles))
    t = torch.zeros(6, width, dtype=torch.int32)
    for g, (slot, pos, rows) in enumerate(segments):
        t[0, g], t[1, g], t[2, g], t[3, g] = slot, pos, rows, row0
        row0 += rows
    for i, (g, r) in enumerate(tiles):
        t[4, i], t[5, i] = g, r
    return t.cuda(), len(tiles)



def test_gemma4_kernels():
    from tensorfold.cuda.sm70.gemma4_native import extension
    ext = extension()
    dev = torch.device("cuda")
    torch.manual_seed(0)
    # w4_rows
    for (M, N, K) in [(1, 1024, 2816), (5, 2816, 512), (16, 272, 2816), (3, 2816, 272)]:
        codes, scales, rowmul = rand_packed(N, K)
        x = torch.randn(M, K)
        out = torch.empty(M, N, device=dev)
        ext.w4_rows(x.to(dev), M, codes.to(dev), scales.to(dev), rowmul.to(dev), N, K, out, N, False)
        check(f"w4_rows M{M} N{N} K{K}", out, x.double() @ dequant(codes, scales, rowmul).T, 1e-5)
    # geglu
    M, W, K = 4, 272, 2816
    codes, scales, rowmul = rand_packed(2 * W, K)
    x = torch.randn(M, K)
    out = torch.empty(M, W, device=dev)
    ext.w4_rows(x.to(dev), M, codes.to(dev), scales.to(dev), rowmul.to(dev), W, K, out, W, True)
    y = x.double() @ dequant(codes, scales, rowmul).T
    check("w4_rows geglu", out, torch.nn.functional.gelu(y[:, :W], approximate="tanh") * y[:, W:], 1e-5)
    # dequant exactness
    w16 = torch.empty(N := 64, K, dtype=torch.float16, device=dev)
    codes, scales, rowmul = rand_packed(N, K)
    ext.w4_dequant(codes.to(dev), scales.to(dev), N, K, w16)
    check("dequant exact", w16.double() * rowmul.double().to(dev)[:, None], dequant(codes, scales, rowmul), 1e-12)
    # experts
    E, W, D, M, S = 8, 176, 2816, 3, 8
    gu = [rand_packed(2 * W, D) for _ in range(E)]; dn = [rand_packed(D, W) for _ in range(E)]
    st = lambda parts, i: torch.stack([p[i] for p in parts]).to(dev).contiguous()
    local = torch.randint(-1, E, (M, S), dtype=torch.int32)
    wts = torch.rand(M, S)
    x = torch.randn(M, D)
    act = torch.empty(M * S, W, device=dev)
    ext.w4_expert_gateup(x.to(dev), M * S, S, local.to(dev), st(gu, 0), st(gu, 1), st(gu, 2), W, D, act)
    out = torch.empty(M, D, device=dev)
    ext.w4_expert_down(act, M, S, local.to(dev), wts.to(dev), st(dn, 0), st(dn, 1), st(dn, 2), D, W, out, D)
    ref = torch.zeros(M, D, dtype=torch.float64)
    for m in range(M):
        for s in range(S):
            e = int(local[m, s])
            if e < 0: continue
            y = x[m].double() @ dequant(*gu[e]).T
            a = torch.nn.functional.gelu(y[:W], approximate="tanh") * y[W:]
            ref[m] += wts[m, s].double() * (a @ dequant(*dn[e]).T)
    check("experts", out, ref, 1e-5)
    # bf16 rows + softcap
    M, N, K = 5, 4096, 2816
    w = torch.randn(N, K).bfloat16(); x = torch.randn(M, K)
    out = torch.empty(M, N, device=dev)
    ext.bf16_rows(x.to(dev), M, w.to(dev), N, K, out, N, 30.0)
    check("bf16_rows softcap", out, torch.tanh((x.double() @ w.double().T) / 30) * 30, 1e-5)
    # i8 rows (DFlash W8A16 proposal copies): K = 704 (down slice), M up to 16
    for M, N, K in [(1, 352, 16896), (5, 2816, 704), (16, 768, 2816)]:
        q8 = torch.randint(-127, 128, (N, K), dtype=torch.int8); sc = torch.rand(N) * 1e-2 + 1e-4
        x = torch.randn(M, K)
        out = torch.empty(M, N, device=dev)
        ext.i8_rows(x.to(dev), M, q8.to(dev), sc.to(dev), N, K, out, N, 0.0)
        check(f"i8_rows M{M} N{N} K{K}", out, (x.double() @ q8.double().T) * sc.double(), 1e-5)
    out = torch.empty(4, 32768, device=dev)                  # proposal head: softcapped INT8 vocabulary rows
    q8 = torch.randint(-127, 128, (32768, 2816), dtype=torch.int8); sc = torch.rand(32768) * 1e-2 + 1e-4
    x = torch.randn(4, 2816)
    ext.i8_rows(x.to(dev), 4, q8.to(dev), sc.to(dev), 32768, 2816, out, 32768, 30.0)
    check("i8_rows softcap head", out, torch.tanh((x.double() @ q8.double().T) * sc.double() / 30) * 30, 1e-5)
    # route
    M = 6
    logits = torch.randn(M, 128) * 3
    scale = (torch.rand(128) + 0.5).bfloat16()
    ids = torch.empty(M, 8, dtype=torch.int32, device=dev); wv = torch.empty(M, 8, device=dev); loc = torch.empty_like(ids)
    ext.route(logits.to(dev), M, 128, 8, scale.to(dev), 64, 64, ids, wv, loc)
    p = torch.softmax(logits.double(), -1); tv, ti = torch.topk(p, 8, -1)
    assert torch.equal(ids.cpu().long(), ti), "route ids"
    check("route weights", wv, tv / tv.sum(-1, keepdim=True) * scale.double()[ti], 1e-6)
    assert torch.equal(loc.cpu(), torch.where(ti >= 64, ti - 64, -1).int()), "route local"
    # attention: R rows x H heads, linear positions, causal (+ window)
    def attn_ref(q, k, v, qpos, kpos, window):
        s = q.double() @ k.double().T
        mask = (kpos[None] >= 0) & (kpos[None] <= qpos[:, None])
        if window: mask &= kpos[None] > qpos[:, None] - window
        s = s.masked_fill(~mask, float("-inf"))
        return torch.softmax(s, -1) @ v.double()
    for D, rows, length, window, ring in [(512, 5, 3000, 0, 0), (256, 1, 1500, 1024, 2048), (256, 40, 2040, 1024, 2048),
                                           (512, 300, 700, 0, 0), (128, 5, 100, 0, 0)]:
        H = 2
        S = ring or (-(-(length) // 256) * 256)
        k = (torch.randn(S, D) * 0.2).half(); v = torch.randn(S, D).half()
        q = torch.randn(rows, H, D).half()
        pos0 = length - rows
        if ring:
            kpos = torch.full((S,), -1, dtype=torch.int32)
            for p_ in range(max(0, length - ring), length): kpos[p_ % ring] = p_
        else:
            kpos = torch.arange(S, dtype=torch.int32)
        qpos = torch.tensor([pos0 + r // H for r in range(rows * H)])
        R = rows * H; rpad = -(-R // 16) * 16; chunks = S // (64 if R <= 16 else 256)
        seg, tiles = segs([(0, pos0, rows)], H)
        kd, vd, pd = k.to(dev), v.to(dev), kpos.to(dev)
        for splits in (1, min(chunks, 7)):
            po = torch.empty(splits, rpad, D, device=dev); pm = torch.empty(splits, rpad, device=dev); pl = torch.empty_like(pm)
            out = torch.empty(R, D, device=dev)
            ext.attn(q.view(R, D).to(dev), H, D, tab(kd), tab(vd), tab(pd) if ring else None, seg, tiles, R <= 16, S,
                     window, 0, 1.0, chunks, splits, po, pm, pl, out)
            ref = attn_ref(q.view(R, D), k, v, qpos, kpos, window)
            check(f"attn D{D} rows{rows} len{length} win{window} splits{splits}", out, ref, 3e-3)
    # qkv_prep: compare with torch reference
    D, Hq, M = 256, 2, 3
    x = torch.randn(M, (Hq + 2) * D)
    qw = (torch.rand(D) + 0.5).bfloat16(); kw = (torch.rand(D) + 0.5).bfloat16()
    inv = (10000.0 ** -(torch.arange(0, D, 2).float() / D))
    kc = torch.zeros(2048, D, dtype=torch.float16, device=dev); vc = torch.zeros_like(kc)
    kp = torch.full((2048,), -1, dtype=torch.int32, device=dev)
    qo = torch.empty(M, Hq, D, dtype=torch.float16, device=dev)
    ext.qkv_prep(x.to(dev), M, Hq, D, 1, qw.to(dev), kw.to(dev), inv.to(dev),
                 torch.arange(2046, 2046 + M, dtype=torch.int32, device=dev), torch.zeros(M, dtype=torch.int32, device=dev),
                 qo, tab(kc), tab(vc), tab(kp), 2048, 1e-6)
    def rope_ref(t, w, pos):
        t = t.double(); t = t / torch.sqrt((t * t).mean(-1, keepdim=True) + 1e-6) * (w.double() if w is not None else 1)
        a, b = t[..., :D // 2], t[..., D // 2:]
        ang = pos * inv.double(); c, s = torch.cos(ang), torch.sin(ang)
        return torch.cat([a * c - b * s, b * c + a * s], -1)
    for m in range(M):
        pos = 2046 + m
        for h in range(Hq):
            check(f"q m{m} h{h}", qo[m, h], rope_ref(x[m, h * D:(h + 1) * D], qw, pos), 2e-3)
        check(f"k m{m}", kc[pos % 2048], rope_ref(x[m, Hq * D:(Hq + 1) * D], kw, pos), 2e-3)
        vv = x[m, (Hq + 1) * D:].double(); vv = vv / torch.sqrt((vv * vv).mean() + 1e-6)
        check(f"v m{m}", vc[pos % 2048], vv, 2e-3)
        assert int(kp[pos % 2048]) == pos


@pytest.mark.parametrize("K", [16, 176, 272, 2816])
@pytest.mark.parametrize("offset", [0, 1, 8])
def test_gemma4_dequant_all_finite_codes_and_output_alignment(K, offset):
    from tensorfold.cuda.sm70.gemma4_native import extension

    # Every signed finite E4M3 scale crossed with all sixteen E2M1 codes,
    # including both signed zeros. Shapes include actual narrow TP slices and
    # non-full launch blocks. An offset of one half requires scalar stores.
    scale_bytes = torch.tensor([b for b in range(256) if b & 0x7f != 0x7f], dtype=torch.uint8)
    N = scale_bytes.numel()
    low = (torch.arange(K // 2) * 2 % 16).to(torch.uint8)
    codes = (low | ((low + 1) << 4)).expand(N, -1).contiguous()
    scales = scale_bytes[:, None].expand(-1, K // 16).contiguous()
    expected = dequant(codes, scales, torch.ones(N)).half()
    storage = torch.full((N * K + offset + 8,), 3.0, dtype=torch.float16, device="cuda")
    out = storage[offset:offset + N * K].view(N, K)
    extension().w4_dequant(codes.cuda(), scales.cuda(), N, K, out)
    assert torch.equal(out.cpu().view(torch.int16), expected.view(torch.int16))
    assert torch.all(storage[:offset] == 3)
    assert torch.all(storage[offset + N * K:] == 3)


@pytest.mark.parametrize("rows", [17, 1024])
def test_gemma4_expert_prompt_combine_preserves_slot_arithmetic(rows):
    from tensorfold.cuda.sm70.gemma4_native import extension

    torch.manual_seed(43)
    dev = torch.device("cuda")
    D, slots, experts, width = 2816, 8, 64, 8
    padded_rows = experts * width
    y = torch.randn(padded_rows, D, device=dev)
    dest = torch.randint(-1, padded_rows, (rows, slots), device=dev)
    weights = torch.randn(rows, slots, device=dev)
    rowmul = torch.randn(experts, D, device=dev)
    # All absent slots must multiply positive zero, including negative and nonfinite
    # weights. The first product, rather than a +0 accumulator, preserves -0 results.
    dest[:2].fill_(-1)
    weights[0].fill_(-1)
    weights[1].fill_(1)
    weights[1, 3] = float("inf")
    # A multiply-add contraction would produce a nonzero residual in this column.
    dest[2].fill_(-1)
    dest[2, 0], dest[2, 1] = 0, 1
    weights[2].zero_()
    weights[2, 0], weights[2, 1] = 1, 1 - 2 ** -23
    y[0, 0], y[1, 0] = -1, 1 + 2 ** -23
    rowmul[0].fill_(1)
    # Tiny values cover gradual underflow as well as ordinary finite results.
    y[2].fill_(torch.finfo(torch.float32).tiny / 2)
    dest[3].fill_(2)
    weights[3].fill_(0.5)
    # Adversarial scales are applied only to real pairs, in a separate rounded
    # multiply before route weights. Preserve NaN behavior and signed zeros.
    rowmul[-1, :4] = torch.tensor([float("nan"), float("inf"), -0.0, -1.0], device=dev)
    dest[4].fill_(padded_rows - 1)
    y[-1, :4] = torch.tensor([1.0, 0.0, 1.0, 0.0], device=dev)
    local = torch.where(dest < 0, -1, dest // width).to(torch.int32)

    # Original production expression is the independent arithmetic/order oracle.
    scaled = (y.view(experts, width, D) * rowmul[:, None, :]).view(padded_rows, D)
    pairs = torch.zeros((rows * slots, D), device=dev)
    flat = dest.reshape(-1)
    present = flat >= 0
    pairs[present] = scaled[flat[present]]
    pairs = pairs.view(rows, slots, D)
    expected = pairs[:, 0] * weights[:, 0:1]
    for s in range(1, slots):
        expected = expected + pairs[:, s] * weights[:, s:s + 1]

    storage = torch.full((rows, 2, D), 123.0, device=dev)
    out = storage[:, 1]
    extension().expert_prompt_combine(y, dest, weights, local, rowmul, out)
    assert torch.equal(torch.isnan(out), torch.isnan(expected))
    finite = ~torch.isnan(expected)
    assert torch.equal(out.view(torch.int32)[finite], expected.view(torch.int32)[finite])
    assert torch.all(storage[:, 0] == 123)


@pytest.mark.parametrize("rows,distribution", [(17, "empty"), (25, "uneven"), (31, "one_group"),
                                               (63, "uneven"), (64, "uneven"), (65, "one_group"),
                                               (129, "uneven"), (1024, "balanced")])
@torch.inference_mode()
def test_gemma4_prompt_expert_groups_match_uniform_bmm(rows, distribution):
    from types import SimpleNamespace
    from tensorfold.cuda.sm70.gemma4_native import extension
    from tensorfold.families.gemma4.sm70.target import Target

    torch.manual_seed(79)
    device = torch.device("cuda")
    E, D, W, slots = 64, 2816, 176, 8
    target = Target.__new__(Target)
    target.c = SimpleNamespace(hidden=D, top_k=slots)
    target.w = SimpleNamespace(expert_width=W, expert_count=E)
    target.device, target.ext = device, extension()
    target._scratch, target.frozen = {}, False

    def packed(n, k):
        return SimpleNamespace(codes=torch.randint(0, 256, (E, n, k // 2), dtype=torch.uint8, device=device),
                               scales=torch.randint(0x20, 0x38, (E, n, k // 16), dtype=torch.uint8, device=device),
                               rowmul=torch.rand(E, n, device=device) * 1e-3 + 1e-4)

    eg, ed = packed(2 * W, D), packed(D, W)
    layer = SimpleNamespace(experts_gate_up=eg, experts_down=ed)
    x = torch.randn(rows, D, device=device).half()
    local = (torch.arange(rows * slots, device=device).reshape(rows, slots) % E).int()
    if distribution == "empty":
        local.fill_(-1)
    elif distribution == "one_group":
        local.copy_(torch.arange(slots, device=device).expand(rows, slots))
    elif distribution == "uneven":
        local[::3] = torch.arange(0, E, slots, dtype=torch.int32, device=device)
        local[::7].fill_(-1)
    weights = torch.rand(rows, slots, device=device)

    # Retained original uniform-padding BMM expression, with independently built
    # expert lists: an oracle for packing, group dispatch and FP32 accumulation.
    flat = local.reshape(-1)
    pair_ids = [torch.nonzero(flat == expert).flatten() for expert in range(E)]
    width = max(1, max(ids.numel() for ids in pair_ids))
    padded = torch.zeros(E, width, D, dtype=torch.float16, device=device)
    for expert, ids in enumerate(pair_ids):
        padded[expert, :ids.numel()] = x[ids // slots]
    gu16 = torch.empty(E, 2 * W, D, dtype=torch.float16, device=device)
    dn16 = torch.empty(E, D, W, dtype=torch.float16, device=device)
    target.ext.w4_dequant(eg.codes, eg.scales, E * 2 * W, D, gu16)
    target.ext.w4_dequant(ed.codes, ed.scales, E * D, W, dn16)
    gu = torch.bmm(padded, gu16.transpose(1, 2), out_dtype=torch.float32).mul_(eg.rowmul[:, None, :])
    act = (torch.nn.functional.gelu(gu[..., :W], approximate="tanh") * gu[..., W:]).half()
    y = torch.bmm(act, dn16.transpose(1, 2), out_dtype=torch.float32).mul_(ed.rowmul[:, None, :])
    pairs = torch.zeros(rows * slots, D, device=device)
    for expert, ids in enumerate(pair_ids):
        pairs[ids] = y[expert, :ids.numel()]
    pairs = pairs.view(rows, slots, D)
    expected = pairs[:, 0] * weights[:, 0:1]
    for slot in range(1, slots):
        expected = expected + pairs[:, slot] * weights[:, slot:slot + 1]
    storage = torch.full((rows, 2, D), 123.0, device=device)
    target._experts_prompt(layer, x, local, weights, storage[:, 1], rows)
    assert torch.isfinite(expected).all()
    assert torch.equal(storage[:, 1].view(torch.int32), expected.view(torch.int32)), distribution
    assert torch.all(storage[:, 0] == 123)


@pytest.mark.parametrize("bits", [8, 16])
@pytest.mark.parametrize("layers,rows", [(2, 5), (5, 1), (5, 5), (5, 16), (5, 17), (5, 1024)])
@torch.inference_mode()
def test_gemma4_dflash_context_kv_bank_and_strided_cache(bits, layers, rows):
    from tensorfold.cuda.sm70.gemma4_native import extension
    from tensorfold.families.gemma4.sm70.dflash import DFlash

    torch.manual_seed(91)
    device = torch.device("cuda")
    D, H, d = 2816, 4, 128
    draft = DFlash.__new__(DFlash)
    draft.bits, draft.heads, draft.d = bits, H, d
    draft.device, draft.ext, draft._f32 = device, extension(), {}
    draft.layers = []
    for _ in range(layers):
        w = (torch.randn((H + 2) * d, D, device=device) * 0.05).bfloat16()
        draft.layers.append({"qkv": draft._int8(w) if bits == 8 else w})
    bank = draft._pack_context_kv()
    tails = [draft._rows(layer["qkv"], H * d) for layer in draft.layers]
    if bits == 8:
        assert torch.equal(bank[0], torch.cat([w[0] for w in tails]))
        assert torch.equal(bank[1], torch.cat([w[1] for w in tails]))
        explicit = bank[0].float() * bank[1][:, None]
        assert torch.equal(draft._w32("kv", bank).view(torch.int32), explicit.view(torch.int32))
    else:
        assert torch.equal(bank, torch.cat(tails))
    x = torch.randn(rows, D, device=device)
    if rows > 16:
        # Known-answer sparse inputs isolate packing/stride from allowed changes
        # in the FP32 library GEMM reduction algorithm when output width changes.
        x.zero_()
        x[torch.arange(rows, device=device), torch.arange(rows, device=device) % D] = 1.0
    projected = draft._linear(x, bank, "kv", rows > 16)
    original = [draft._linear(x, weight, f"reference-kv{i}", rows > 16) for i, weight in enumerate(tails)]
    assert torch.equal(projected.view(torch.int32), torch.cat(original, dim=1).view(torch.int32))
    before = projected.clone()
    pos = torch.arange(19, 19 + rows, dtype=torch.int32, device=device)
    slot = torch.zeros(rows, dtype=torch.int32, device=device)
    inv = (1.0 / (10000 ** (torch.arange(0, d, 2, device=device).float() / d))).contiguous()
    qw = torch.ones(d, dtype=torch.bfloat16, device=device)
    for i in range(layers):
        kw = (torch.rand(d, device=device) + 0.5).bfloat16()
        k = torch.full((rows + 32, d), 123.0, dtype=torch.float16, device=device)
        v = torch.full_like(k, 124.0)
        expected_k, expected_v = k.clone(), v.clone()
        part = projected[:, i * 2 * d:(i + 1) * 2 * d]
        draft.ext.qkv_prep(part, rows, 0, d, 2, qw, kw, inv, pos, slot, part,
                           tab(k), tab(v), None, 0, 1e-6)
        draft.ext.qkv_prep(original[i], rows, 0, d, 2, qw, kw, inv, pos, slot, original[i],
                           tab(expected_k), tab(expected_v), None, 0, 1e-6)
        assert torch.equal(k.view(torch.int16), expected_k.view(torch.int16))
        assert torch.equal(v.view(torch.int16), expected_v.view(torch.int16))
        assert torch.all(k[:19] == 123) and torch.all(k[19 + rows:] == 123)
        assert torch.all(v[:19] == 124) and torch.all(v[19 + rows:] == 124)
    assert torch.equal(projected.view(torch.int32), before.view(torch.int32))  # Hq0 never writes qout


def test_gemma4_dflash_int8_prompt_promotion_matches_explicit_float():
    from tensorfold.families.gemma4.sm70.dflash import DFlash

    values = torch.arange(-128, 128, dtype=torch.int16, device="cuda").to(torch.int8)
    scales = torch.tensor([0.0, -0.0, torch.finfo(torch.float32).tiny / 2,
                           torch.finfo(torch.float32).tiny, 2 ** -20, 1.0, 2 ** 20,
                           torch.finfo(torch.float32).max / 128], device="cuda")
    q = values.expand(scales.numel(), -1).contiguous()
    draft = DFlash.__new__(DFlash)
    draft._f32 = {}
    actual = draft._w32("promotion", (q, scales))
    expected = q.float() * scales[:, None]
    assert actual.dtype == torch.float32
    assert torch.equal(actual.view(torch.int32), expected.view(torch.int32))


@pytest.mark.parametrize("outputs,inputs,softcap", [(32768, 2816, 30.0), (2816, 512, 0.0),
                                                   (1408, 2816, 0.0), (2816, 704, 0.0),
                                                   (1280, 2816, 0.0), (1025, 32, 30.0),
                                                   (768, 2816, 0.0)])
def test_gemma4_bf16_exact_five_matches_unchanged_row_controls(outputs, inputs, softcap):
    from tensorfold.cuda.sm70.gemma4_native import extension

    torch.manual_seed(125)
    ext = extension()
    device = torch.device("cuda")
    weights = (torch.randn(outputs, inputs, device=device) * 0.05).bfloat16()
    storage = torch.randn(6, inputs + 8, device=device)
    x6 = storage[:, :inputs]
    reference = torch.empty(6, outputs, device=device)
    # M6 is the unchanged MAXM8 path; M4 is an adjacent unchanged MAXM4 control.
    # The declared native per-row arithmetic does not depend on row count.
    ext.bf16_rows(x6, 6, weights, outputs, inputs, reference, outputs, softcap)
    for rows in (4, 5):
        x_storage = torch.empty(rows, inputs + 8, device=device)
        x = x_storage[:, :inputs]
        x.copy_(x6[:rows])
        guarded = torch.full((rows + 2, outputs + 8), 123.0, device=device)
        out = guarded[1:rows + 1, :outputs]
        ext.bf16_rows(x, rows, weights, outputs, inputs, out, guarded.stride(0), softcap)
        assert torch.equal(out.view(torch.int32), reference[:rows].view(torch.int32))
        assert torch.all(guarded[0] == 123) and torch.all(guarded[-1] == 123)
        assert torch.all(guarded[1:rows + 1, outputs:] == 123)


def test_gemma4_fused_router_matches_logits_and_routing():
    from tensorfold.cuda.sm70.gemma4_native import extension
    ext = extension()
    dev = torch.device("cuda")
    torch.manual_seed(1)
    E, K = 128, 2816
    w = (torch.randn(E, K) * 0.05).bfloat16()
    scale = (torch.rand(E) + 0.5).bfloat16()
    counter = torch.zeros(20, dtype=torch.int32, device=dev)   # one per two-row router tile
    for M in (1, 5, 16, 40):
        x = torch.randn(M, K)
        logits = torch.empty(M, E, device=dev)
        ids = torch.empty(M, 8, dtype=torch.int32, device=dev)
        wv = torch.empty(M, 8, device=dev)
        loc = torch.empty_like(ids)
        for _ in range(2):                       # the last block resets the counter for the next launch
            ext.router(x.to(dev), M, w.to(dev), E, K, logits, counter, 8, scale.to(dev), 64, 64, ids, wv, loc)
        ref = x.double() @ w.double().T
        check(f"router logits M{M}", logits, ref, 1e-5)
        p = torch.softmax(logits.double().cpu(), -1)
        tv, ti = torch.topk(p, 8, -1)
        assert torch.equal(ids.cpu().long(), ti)
        check(f"router weights M{M}", wv, tv / tv.sum(-1, keepdim=True) * scale.double()[ti], 1e-6)
        assert torch.equal(loc.cpu(), torch.where(ti >= 64, ti - 64, -1).int())
        assert int(counter.abs().sum().item()) == 0


@pytest.mark.parametrize("rows", [17, 33, 160])
def test_gemma4_row_tiles_equal_sixteen_row_calls(rows):
    """Calls over 16 rows run 16-row tiles on grid.y with each row's arithmetic unchanged (bitwise)."""
    from tensorfold.cuda.sm70.gemma4_native import extension
    ext = extension()
    dev = torch.device("cuda")
    torch.manual_seed(7)
    for N, K, geglu in [(1024, 2816, False), (2816, 512, False), (272, 2816, True), (2816, 272, False)]:
        codes, scales, rowmul = (t.to(dev) for t in rand_packed(2 * N if geglu else N, K))
        x = torch.randn(rows, K, device=dev)
        whole = torch.empty(rows, N, device=dev)
        ext.w4_rows(x, rows, codes, scales, rowmul, N, K, whole, N, geglu)
        for r0 in range(0, rows, 16):
            m = min(16, rows - r0)
            part = torch.empty(m, N, device=dev)
            ext.w4_rows(x[r0:r0 + m], m, codes, scales, rowmul, N, K, part, N, geglu)
            assert torch.equal(whole[r0:r0 + m].view(torch.int32), part.view(torch.int32))
    w = (torch.randn(3000, 2816, device=dev) * 0.05).bfloat16()
    q8 = torch.randint(-127, 128, (1408, 2816), dtype=torch.int8, device=dev); sc = torch.rand(1408, device=dev) * 1e-2
    x = torch.randn(rows, 2816, device=dev)
    for name, call, N in [("bf16", lambda xx, m, o: ext.bf16_rows(xx, m, w, 3000, 2816, o, 3000, 30.0), 3000),
                          ("i8", lambda xx, m, o: ext.i8_rows(xx, m, q8, sc, 1408, 2816, o, 1408, 0.0), 1408)]:
        whole = torch.empty(rows, N, device=dev)
        call(x, rows, whole)
        for r0 in range(0, rows, 16):
            m = min(16, rows - r0)
            part = torch.empty(m, N, device=dev)
            call(x[r0:r0 + m].contiguous(), m, part)
            assert torch.equal(whole[r0:r0 + m].view(torch.int32), part.view(torch.int32)), name


@pytest.mark.parametrize("rows", [2, 3, 5, 6, 8, 10, 12, 16, 20, 40])
def test_gemma4_rows_equal_one_row_calls(rows):
    """Every row of a multi-row decode call (any row tile) equals its own one-row call (bitwise): W4 projections
    and GEGLU, the BF16 head, DFlash INT8 rows and the fused router."""
    from tensorfold.cuda.sm70.gemma4_native import extension
    ext = extension()
    dev = torch.device("cuda")
    torch.manual_seed(13)

    def same(name, call, x, n):
        whole = torch.empty(rows, n, device=dev)
        call(x, rows, whole)
        one = torch.empty(1, n, device=dev)
        for r in range(rows):
            call(x[r:r + 1], 1, one)
            assert torch.equal(whole[r].view(torch.int32), one[0].view(torch.int32)), f"{name} row {r} of {rows}"

    for N, K, geglu in [(1024, 2816, False), (1536, 2816, False), (2816, 512, False), (272, 2816, True),
                        (2816, 272, False)]:
        codes, scales, rowmul = (t.to(dev) for t in rand_packed(2 * N if geglu else N, K))
        same(f"w4 {N}x{K}", lambda xx, m, o: ext.w4_rows(xx, m, codes, scales, rowmul, N, K, o, N, geglu),
             torch.randn(rows, K, device=dev), N)
    head = (torch.randn(4096, 2816, device=dev) * 0.05).bfloat16()
    same("bf16", lambda xx, m, o: ext.bf16_rows(xx, m, head, 4096, 2816, o, 4096, 30.0),
         torch.randn(rows, 2816, device=dev), 4096)
    for N, K in [(768, 2816), (2816, 512), (1408, 2816), (2816, 704), (32768, 2816)]:
        q8 = torch.randint(-127, 128, (N, K), dtype=torch.int8, device=dev); sc = torch.rand(N, device=dev) * 1e-2
        same(f"i8 {N}x{K}", lambda xx, m, o: ext.i8_rows(xx, m, q8, sc, N, K, o, N, 0.0),
             torch.randn(rows, K, device=dev), N)
    E, K = 128, 2816
    w = (torch.randn(E, K, device=dev) * 0.05).bfloat16(); scale = (torch.rand(E, device=dev) + 0.5).bfloat16()
    counter = torch.zeros(64, dtype=torch.int32, device=dev)
    x = torch.randn(rows, K, device=dev)

    def route(xx, m):
        out = (torch.empty(m, E, device=dev), torch.empty(m, 8, dtype=torch.int32, device=dev),
               torch.empty(m, 8, device=dev), torch.empty(m, 8, dtype=torch.int32, device=dev))
        ext.router(xx, m, w, E, K, out[0], counter, 8, scale, 64, 64, out[1], out[2], out[3])
        return out

    whole = route(x, rows)
    for r in range(rows):
        one = route(x[r:r + 1], 1)
        for a, b in zip(whole, one):
            assert torch.equal(a[r].view(torch.int32), b[0].view(torch.int32)), f"router row {r} of {rows}"
    assert int(counter.abs().sum().item()) == 0


@pytest.mark.parametrize("D,n,ring", [(512, 1, 0), (512, 5, 0), (256, 5, 2048), (128, 5, 0), (256, 300, 2048)])
def test_gemma4_attention_segments_equal_single_stream_calls(D, n, ring):
    """Several streams' segments (own slots/positions) in one launch equal each stream's own launch (bitwise)."""
    from tensorfold.cuda.sm70.gemma4_native import extension
    ext = extension()
    dev = torch.device("cuda")
    torch.manual_seed(11)
    H = 4 if D == 128 else 2
    lengths = [700, 3000, 64, 1500]
    S = ring or 3072
    ks = [(torch.randn(S, D, device=dev) * 0.2).half() for _ in lengths]
    vs = [torch.randn(S, D, device=dev).half() for _ in lengths]
    ps = []
    for L in lengths:
        kpos = torch.full((S,), -1, dtype=torch.int32, device=dev)
        if ring:
            for p_ in range(max(0, L + n - ring), L + n):
                kpos[p_ % ring] = p_
        ps.append(kpos)
    q = torch.randn(len(lengths) * n, H, D, device=dev).half()
    small = n * H <= 16
    chunks = S // (64 if small else 256)
    splits = min(chunks, 9)
    window = 1024 if ring else 0
    segments = [(g, L, n) for g, L in enumerate(lengths)]
    seg, tiles = segs(segments, H)
    po = torch.empty(splits, tiles * 16, D, device=dev); pm = torch.empty(splits, tiles * 16, device=dev)
    pl = torch.empty_like(pm)
    out = torch.empty(len(lengths) * n * H, D, device=dev)
    ext.attn(q.view(-1, D), H, D, tab(*ks), tab(*vs), tab(*ps) if ring else None, seg, tiles, small, S, window,
             0, 1.0, chunks, splits, po, pm, pl, out)
    for g, L in enumerate(lengths):
        one, t1 = segs([(0, L, n)], H)
        po1 = torch.empty(splits, t1 * 16, D, device=dev); pm1 = torch.empty(splits, t1 * 16, device=dev)
        pl1 = torch.empty_like(pm1)
        ref = torch.empty(n * H, D, device=dev)
        ext.attn(q[g * n:(g + 1) * n].reshape(-1, D).contiguous(), H, D, tab(ks[g]), tab(vs[g]),
                 tab(ps[g]) if ring else None, one, t1, small, S, window, 0, 1.0, chunks, splits, po1, pm1, pl1, ref)
        assert torch.equal(out[g * n * H:(g + 1) * n * H].view(torch.int32), ref.view(torch.int32))


@pytest.mark.parametrize("rows", [17, 37, 160])
def test_gemma4_wide_mma_matches_fp64(rows):
    """Wide decode tensor-core projections (hi/lo FP16 activations, exact FP16 weights, FP32 accumulation) and the
    expert-grouped MoE agree with FP64 references of the same contracts to FP32-level relative error."""
    from tensorfold.cuda.sm70.gemma4_native import extension
    ext = extension()
    dev = torch.device("cuda")
    torch.manual_seed(5)
    xh = torch.empty(rows * 2816, dtype=torch.float16, device=dev); xl = torch.empty_like(xh)
    part = torch.empty(rows * 8 * 2 * 2816, device=dev)
    for N, K, geglu in [(1024, 2816, False), (2816, 512, False), (272, 2816, True), (2816, 272, False), (40, 176, False)]:
        codes, scales, rowmul = rand_packed(2 * N if geglu else N, K)
        x = torch.randn(rows, K)
        out = torch.full((rows, N + 3), 7.0, device=dev)
        ext.w4_mma(x.to(dev), rows, codes.to(dev), scales.to(dev), rowmul.to(dev), N, K, out, N + 3, geglu, xh, xl, part)
        y = x.double() @ dequant(codes, scales, rowmul).T
        ref = torch.nn.functional.gelu(y[:, :N], approximate="tanh") * y[:, N:] if geglu else y
        check(f"w4_mma N{N} K{K} geglu{geglu}", out[:, :N], ref, 2e-6)
        assert torch.all(out[:, N:].cpu() == 7.0)
    E, W, D, S = 64, 176, 2816, 8
    gu = [rand_packed(2 * W, D) for _ in range(E)]; dn = [rand_packed(D, W) for _ in range(E)]
    st = lambda parts, i: torch.stack([p[i] for p in parts]).to(dev).contiguous()
    ids = torch.stack([torch.randperm(128)[:8] for _ in range(rows)])
    ids[: rows // 2, 0] = 70                                  # one crowded expert
    local = torch.where(ids >= 64, ids - 64, -1).int()
    wts = torch.rand(rows, S)
    x = torch.randn(rows, D)
    out = torch.full((rows, 2 * D), 7.0, device=dev)
    ext.w4_experts_mma(x.to(dev), rows, S, local.to(dev), wts.to(dev), st(gu, 0), st(gu, 1), st(gu, 2), st(dn, 0),
                       st(dn, 1), st(dn, 2), E, W, D, torch.empty(E, dtype=torch.int32, device=dev),
                       torch.empty(E, rows, dtype=torch.int32, device=dev), torch.empty(rows * S, W, device=dev),
                       torch.empty(rows * S, D, device=dev), out[:, D:], 2 * D, xh, xl, part)
    ref = torch.zeros(rows, D, dtype=torch.float64)
    for m in range(rows):
        for s_ in range(S):
            e = int(local[m, s_])
            if e < 0: continue
            y = x[m].double() @ dequant(*gu[e]).T
            a = torch.nn.functional.gelu(y[:W], approximate="tanh") * y[W:]
            ref[m] += wts[m, s_].double() * (a @ dequant(*dn[e]).T)
    check("experts mma", out[:, D:], ref, 2e-6)
    assert torch.all(out[:, :D].cpu() == 7.0)

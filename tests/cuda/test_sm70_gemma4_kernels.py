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
    assert err < tol, name



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
        R = rows * H; rpad = -(-R // 16) * 16; chunks = S // 256
        for splits in (1, min(chunks, 7)):
            po = torch.empty(splits, rpad, D, device=dev); pm = torch.empty(splits, rpad, device=dev); pl = torch.empty_like(pm)
            out = torch.empty(R, D, device=dev)
            ext.attn(q.view(R, D).to(dev), R, H, D, k.to(dev), v.to(dev), kpos.to(dev) if ring else None, S,
                     torch.tensor([pos0], dtype=torch.int32, device=dev), window, 0, chunks, splits, po, pm, pl, rpad, out)
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
    ext.qkv_prep(x.to(dev), M, Hq, D, 1, qw.to(dev), kw.to(dev), inv.to(dev), torch.tensor([2046], dtype=torch.int32, device=dev),
                 qo, kc, vc, kp, 2048, 1e-6)
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

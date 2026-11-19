"""Gemma 4 SM70 quantized target KV (full-attention 512-wide linear and sliding 256-wide ring caches) (gemma4.cu FMT 8 / 11) against Flash Next's RotorQuant codec.

Independent oracle: the SM70 RotorQuant codec extension (rotorquant_q4_native: encode/rotate/decode). The writer
(qkv_prep) must store exactly the codec's bytes for the same FP32 rows and rotate queries exactly as the codec's
rotation; the reader (attn) must equal FP64 attention over the codec's decoded FP16 operands, rotated back.
"""
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0),
                                reason="needs an SM70 GPU")

FORMATS = [8, 11]


def _row_bytes(fmt, D):
    return D + 4 * (D // 128) if fmt == 8 else D // 2 + 4 * (D // 128)


def _code_bytes(fmt, D):
    return D if fmt == 8 else D // 2


def _split(rows, fmt, D):
    """Cache rows [C, row bytes] -> the codec's (codes [C, 1, code bytes], metadata FP32 words [C, 1, D / 128])."""
    cb = _code_bytes(fmt, D)
    codes = rows[:, :cb].contiguous().view(-1, 1, cb)
    scales = rows[:, cb:].contiguous().view(torch.float32).view(-1, 1, D // 128)
    return codes, scales


def _tab(*tensors):
    return torch.tensor([t.data_ptr() for t in tensors], dtype=torch.int64, device="cuda")


def _segs(segments, H):
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


@pytest.mark.parametrize("fmt", FORMATS)
@pytest.mark.parametrize("D,ring", [(512, 0), (256, 2048)])
def test_writer_stores_codec_bytes_and_rotates_queries(fmt, D, ring):
    """qkv_prep with FMT: K/V rows equal the codec's encoding of the FP32 rows bitwise, queries equal the codec's
    rotation of the FP32 queries bitwise. Head 0 is fed the K input with qw == kw, so its FP32 query output is the
    exact FP32 key the writer encodes; v_mode 2 stores the raw FP32 V input. Ring rows wrap at ``ring``."""
    from tensorfold.cuda.sm70.gemma4_native import extension
    from tensorfold.cuda.sm70 import rotorquant_q4_native as rq
    ext = extension()
    dev = torch.device("cuda")
    torch.manual_seed(5)
    M, Hq = 37, 2
    cap = ring or 512
    base = 2030 if ring else 100
    x = torch.randn(M, (Hq + 2) * D, device=dev)
    x[:, (Hq + 1) * D:] *= torch.rand(M, 1, device=dev) * 30          # V rows over a range of scales
    x[3, (Hq + 1) * D:] = 0                                           # all-zero group rows
    x[:, :D] = x[:, Hq * D:(Hq + 1) * D]                              # head 0 = the key input
    w = (torch.rand(D, device=dev) + 0.5).bfloat16()
    inv = (1e6 ** -(torch.arange(0, D, 2, device=dev).float() / D))
    inv[D // 8:] = 0                                                  # partial rotary as the full layers
    pos = torch.arange(base, base + M, dtype=torch.int32, device=dev)
    rows = (pos.long() % cap) if ring else pos.long()
    slot = torch.zeros(M, dtype=torch.int32, device=dev)
    kp = torch.full((cap,), -1, dtype=torch.int32, device=dev) if ring else None
    q32 = torch.empty(M, Hq, D, device=dev)
    k16 = torch.zeros(cap, D, dtype=torch.float16, device=dev); v16 = torch.zeros_like(k16)
    ext.qkv_prep(x, M, Hq, D, 2, w, w, inv, pos, slot, q32, _tab(k16), _tab(v16), _tab(kp) if ring else None, ring,
                 1e-6, 0)
    rb = _row_bytes(fmt, D)
    kq = torch.zeros(cap, rb, dtype=torch.uint8, device=dev); vq = torch.zeros_like(kq)
    qr = torch.empty(M, Hq, D, device=dev)
    ext.qkv_prep(x, M, Hq, D, 2, w, w, inv, pos, slot, qr, _tab(kq), _tab(vq), _tab(kp) if ring else None, ring,
                 1e-6, fmt)
    if ring:
        assert torch.equal(kp[rows], pos), "ring positions"
    status = torch.tensor([0, -2], dtype=torch.int32, device=dev)
    for name, src, got in (("K", q32[:, 0:1].contiguous(), kq), ("V", x[:, None, (Hq + 1) * D:].contiguous(), vq)):
        codes = torch.zeros(M, 1, _code_bytes(fmt, D), dtype=torch.uint8, device=dev)
        scales = torch.zeros(M, 1, D // 128, device=dev)
        rq.encode(src, codes, scales, codec=fmt, status=status)
        mine_codes, mine_scales = _split(got[rows], fmt, D)
        assert torch.equal(mine_codes, codes), f"{name} codes"
        assert torch.equal(mine_scales.view(torch.int32), scales.view(torch.int32)), f"{name} scales"
        untouched = torch.ones(cap, dtype=torch.bool, device=dev)
        untouched[rows] = False
        assert not got[untouched].any(), f"{name}: rows outside the written positions changed"
    assert int(status[0]) == 0, "codec reported an encoding error"
    rot = torch.empty_like(q32)
    rq.rotate(q32, rot, codec=fmt)
    assert torch.equal(qr.view(torch.int32), rot.view(torch.int32)), "query rotation"


def _fill(fmt, S, length, scale, dev, D=512):
    """A linear cache of S rows: the first `length` rows the codec's encoding of random FP32 K/V, later rows zero.
    Returns (rows, decoded rotated-basis FP16 operands [S, D])."""
    src = torch.randn(length, 1, D, device=dev) * scale
    codes = torch.zeros(S, 1, _code_bytes(fmt, D), dtype=torch.uint8, device=dev)
    scales = torch.zeros(S, 1, D // 128, device=dev)
    from tensorfold.cuda.sm70 import rotorquant_q4_native as rq
    rq.encode(src, codes, scales, codec=fmt, rows=length)
    rows = torch.cat([codes.view(S, -1), scales.view(S, -1).view(torch.uint8)], 1).contiguous()
    dec = torch.empty(S, 1, D, dtype=torch.float16, device=dev)
    rq.decode(codes, scales, dec, codec=fmt)
    return rows, dec.view(S, D)


@pytest.mark.parametrize("fmt", FORMATS)
@pytest.mark.parametrize("D,ring,rows,length", [(512, 0, 1, 3000), (512, 0, 5, 3000), (512, 0, 6, 700),
                                                (512, 0, 300, 700), (512, 0, 1024, 1536), (256, 2048, 1, 1500),
                                                (256, 2048, 8, 5000), (256, 2048, 40, 2040), (256, 2048, 1024, 3000)])
def test_attention_matches_fp64_over_decoded_operands(fmt, D, ring, rows, length):
    """Causal (ring: 1,024-window) attention over quantized K/V equals FP64 attention of the rotated query against
    the codec's decoded FP16 K/V, rotated back by the codec (FP16 operands, FP32 accumulation: the FP16 cache
    test's bound)."""
    from tensorfold.cuda.sm70.gemma4_native import extension
    from tensorfold.cuda.sm70 import rotorquant_q4_native as rq
    ext = extension()
    dev = torch.device("cuda")
    torch.manual_seed(7 + rows)
    H = 2
    S = ring or -(-(length + 16) // 256) * 256
    window = 1024 if ring else 0
    kr, kd = _fill(fmt, S, S if ring else length, 0.2, dev, D)
    vr, vd = _fill(fmt, S, S if ring else length, 1.0, dev, D)
    if ring:
        kpos = torch.full((S,), -1, dtype=torch.int32, device=dev)
        for p_ in range(max(0, length - ring), length):
            kpos[p_ % ring] = p_
    else:
        kpos = torch.arange(S, dtype=torch.int32, device=dev)
    q = torch.randn(rows, H, D, device=dev)
    qrot = torch.empty_like(q)
    rq.rotate(q, qrot, codec=fmt)
    q16 = qrot.half()
    pos0 = length - rows
    R = rows * H
    small = R <= 16
    chunks = S // (64 if small else 256)
    seg, tiles = _segs([(0, pos0, rows)], H)
    qpos = torch.tensor([pos0 + r // H for r in range(R)], device=dev)
    s = q16.view(R, D).double() @ kd.double().T
    mask = (kpos[None] >= 0) & (kpos[None] <= qpos[:, None])
    if window:
        mask &= kpos[None] > qpos[:, None] - window
    ref_rot = torch.softmax(s.masked_fill(~mask, float("-inf")), -1) @ vd.double()
    ref = torch.empty(R, 1, D, device=dev)
    rq.rotate(ref_rot.float().view(R, 1, D), ref, codec=fmt, inverse=True)
    for splits in (1, min(chunks, 7)):
        po = torch.empty(splits, tiles * 16, D, device=dev)
        pm = torch.empty(splits, tiles * 16, device=dev); pl = torch.empty_like(pm)
        out = torch.empty(R, D, device=dev)
        ext.attn(q16.view(R, D), H, D, _tab(kr), _tab(vr), _tab(kpos) if ring else None, seg, tiles, small, S,
                 window, 0, 1.0, chunks, splits, po, pm, pl, out, fmt)
        err = ((out.double() - ref.view(R, D).double()).abs().max() / ref.double().abs().max()).item()
        assert err < 3e-3, f"splits {splits}: relative error {err:.3g}"


@pytest.mark.parametrize("fmt", FORMATS)
@pytest.mark.parametrize("n", [1, 5, 300])
def test_attention_segments_equal_single_stream_calls(fmt, n):
    """Several streams' segments in one quantized launch equal each stream's own launch (bitwise)."""
    from tensorfold.cuda.sm70.gemma4_native import extension
    ext = extension()
    dev = torch.device("cuda")
    torch.manual_seed(13)
    H, S, D = 2, 3072, 512
    lengths = [700, 3000, 64, 1500]
    caches = [(_fill(fmt, S, S, 0.2, dev)[0], _fill(fmt, S, S, 1.0, dev)[0]) for _ in lengths]
    q = torch.randn(len(lengths) * n, H, D, device=dev).half()
    small = n * H <= 16
    chunks = S // (64 if small else 256)
    splits = min(chunks, 9)
    seg, tiles = _segs([(g, L, n) for g, L in enumerate(lengths)], H)
    po = torch.empty(splits, tiles * 16, D, device=dev); pm = torch.empty(splits, tiles * 16, device=dev)
    pl = torch.empty_like(pm)
    out = torch.empty(len(lengths) * n * H, D, device=dev)
    ext.attn(q.view(-1, D), H, D, _tab(*(k for k, _ in caches)), _tab(*(v for _, v in caches)), None, seg, tiles,
             small, S, 0, 0, 1.0, chunks, splits, po, pm, pl, out, fmt)
    for g, L in enumerate(lengths):
        one, t1 = _segs([(0, L, n)], H)
        po1 = torch.empty(splits, t1 * 16, D, device=dev); pm1 = torch.empty(splits, t1 * 16, device=dev)
        pl1 = torch.empty_like(pm1)
        ref = torch.empty(n * H, D, device=dev)
        ext.attn(q[g * n:(g + 1) * n].reshape(-1, D).contiguous(), H, D, _tab(caches[g][0]), _tab(caches[g][1]),
                 None, one, t1, small, S, 0, 0, 1.0, chunks, splits, po1, pm1, pl1, ref, fmt)
        assert torch.equal(out[g * n * H:(g + 1) * n * H].view(torch.int32), ref.view(torch.int32))


def test_formats_refuse_unsupported_caches():
    """Quantized attention serves only the target's cache shapes (linear 512, ring 256)."""
    from tensorfold.cuda.sm70.gemma4_native import extension
    ext = extension()
    dev = torch.device("cuda")
    D, S = 128, 256
    k = torch.zeros(S, _row_bytes(8, D), dtype=torch.uint8, device=dev)
    q = torch.zeros(2, D, dtype=torch.float16, device=dev)
    seg, tiles = _segs([(0, 0, 1)], 2)
    po = torch.empty(1, 16, D, device=dev); pm = torch.empty(1, 16, device=dev); pl = torch.empty_like(pm)
    out = torch.empty(2, D, device=dev)
    with pytest.raises(RuntimeError, match="linear 512 or ring 256"):
        ext.attn(q, 2, D, _tab(k), _tab(k), None, seg, tiles, True, S, 0, 0, 1.0, S // 64, 1, po, pm, pl, out, 8)

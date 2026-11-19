"""Gemma 4 SM70 quantized full-attention KV (gemma4.cu FMT 8 / 11) against Flash Next's RotorQuant codec.

Independent oracle: the SM70 RotorQuant codec extension (rotorquant_q4_native: encode/rotate/decode). The writer
(qkv_prep) must store exactly the codec's bytes for the same FP32 rows and rotate queries exactly as the codec's
rotation; the reader (attn) must equal FP64 attention over the codec's decoded FP16 operands, rotated back.
"""
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0),
                                reason="needs an SM70 GPU")

D = 512
FORMATS = [8, 11]


def _row_bytes(fmt):
    return D + 4 * (D // 128) if fmt == 8 else D // 2 + 4 * (D // 128)


def _code_bytes(fmt):
    return D if fmt == 8 else D // 2


def _split(rows, fmt):
    """Cache rows [C, row bytes] -> the codec's (codes [C, 1, code bytes], metadata FP32 words [C, 1, D / 128])."""
    cb = _code_bytes(fmt)
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
def test_writer_stores_codec_bytes_and_rotates_queries(fmt):
    """qkv_prep with FMT: K/V rows equal the codec's encoding of the FP32 rows bitwise, queries equal the codec's
    rotation of the FP32 queries bitwise. Head 0 is fed the K input with qw == kw, so its FP32 query output is the
    exact FP32 key the writer encodes; v_mode 2 stores the raw FP32 V input."""
    from tensorfold.cuda.sm70.gemma4_native import extension
    from tensorfold.cuda.sm70 import rotorquant_q4_native as rq
    ext = extension()
    dev = torch.device("cuda")
    torch.manual_seed(5)
    M, Hq, cap = 37, 2, 512
    x = torch.randn(M, (Hq + 2) * D, device=dev)
    x[:, (Hq + 1) * D:] *= torch.rand(M, 1, device=dev) * 30          # V rows over a range of scales
    x[3, (Hq + 1) * D:] = 0                                           # all-zero group rows
    x[:, :D] = x[:, Hq * D:(Hq + 1) * D]                              # head 0 = the key input
    w = (torch.rand(D, device=dev) + 0.5).bfloat16()
    inv = (1e6 ** -(torch.arange(0, D, 2, device=dev).float() / D))
    inv[D // 8:] = 0                                                  # partial rotary as the full layers
    pos = torch.arange(100, 100 + M, dtype=torch.int32, device=dev)
    slot = torch.zeros(M, dtype=torch.int32, device=dev)
    q32 = torch.empty(M, Hq, D, device=dev)
    k16 = torch.zeros(cap, D, dtype=torch.float16, device=dev); v16 = torch.zeros_like(k16)
    ext.qkv_prep(x, M, Hq, D, 2, w, w, inv, pos, slot, q32, _tab(k16), _tab(v16), None, 0, 1e-6, 0)
    rb = _row_bytes(fmt)
    kq = torch.zeros(cap, rb, dtype=torch.uint8, device=dev); vq = torch.zeros_like(kq)
    qr = torch.empty(M, Hq, D, device=dev)
    ext.qkv_prep(x, M, Hq, D, 2, w, w, inv, pos, slot, qr, _tab(kq), _tab(vq), None, 0, 1e-6, fmt)
    status = torch.tensor([0, -2], dtype=torch.int32, device=dev)
    for name, src, got in (("K", q32[:, 0:1].contiguous(), kq), ("V", x[:, None, (Hq + 1) * D:].contiguous(), vq)):
        codes = torch.zeros(M, 1, _code_bytes(fmt), dtype=torch.uint8, device=dev)
        scales = torch.zeros(M, 1, D // 128, device=dev)
        rq.encode(src, codes, scales, codec=fmt, status=status)
        mine_codes, mine_scales = _split(got[100:100 + M], fmt)
        assert torch.equal(mine_codes, codes), f"{name} codes"
        assert torch.equal(mine_scales.view(torch.int32), scales.view(torch.int32)), f"{name} scales"
    assert int(status[0]) == 0, "codec reported an encoding error"
    assert not kq[:100].any() and not kq[100 + M:].any(), "rows outside the written positions changed"
    rot = torch.empty_like(q32)
    rq.rotate(q32, rot, codec=fmt)
    assert torch.equal(qr.view(torch.int32), rot.view(torch.int32)), "query rotation"


def _fill(fmt, S, length, scale, dev):
    """A linear cache of S rows: the first `length` rows the codec's encoding of random FP32 K/V, later rows zero.
    Returns (rows, decoded rotated-basis FP16 operands [S, D])."""
    src = torch.randn(length, 1, D, device=dev) * scale
    codes = torch.zeros(S, 1, _code_bytes(fmt), dtype=torch.uint8, device=dev)
    scales = torch.zeros(S, 1, D // 128, device=dev)
    from tensorfold.cuda.sm70 import rotorquant_q4_native as rq
    rq.encode(src, codes, scales, codec=fmt, rows=length)
    rows = torch.cat([codes.view(S, -1), scales.view(S, -1).view(torch.uint8)], 1).contiguous()
    dec = torch.empty(S, 1, D, dtype=torch.float16, device=dev)
    rq.decode(codes, scales, dec, codec=fmt)
    return rows, dec.view(S, D)


@pytest.mark.parametrize("fmt", FORMATS)
@pytest.mark.parametrize("rows,length", [(1, 3000), (5, 3000), (6, 700), (300, 700), (1024, 1536)])
def test_attention_matches_fp64_over_decoded_operands(fmt, rows, length):
    """Causal attention over quantized K/V equals FP64 attention of the rotated query against the codec's decoded
    FP16 K/V, rotated back by the codec (FP16 operands, FP32 accumulation: same bound as the FP16 cache test)."""
    from tensorfold.cuda.sm70.gemma4_native import extension
    from tensorfold.cuda.sm70 import rotorquant_q4_native as rq
    ext = extension()
    dev = torch.device("cuda")
    torch.manual_seed(7 + rows)
    H = 2
    S = -(-(length + 16) // 256) * 256
    kr, kd = _fill(fmt, S, length, 0.2, dev)
    vr, vd = _fill(fmt, S, length, 1.0, dev)
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
    mask = torch.arange(S, device=dev)[None] <= qpos[:, None]
    ref_rot = torch.softmax(s.masked_fill(~mask, float("-inf")), -1) @ vd.double()
    ref = torch.empty(R, 1, D, device=dev)
    rq.rotate(ref_rot.float().view(R, 1, D), ref, codec=fmt, inverse=True)
    for splits in (1, min(chunks, 7)):
        po = torch.empty(splits, tiles * 16, D, device=dev)
        pm = torch.empty(splits, tiles * 16, device=dev); pl = torch.empty_like(pm)
        out = torch.empty(R, D, device=dev)
        ext.attn(q16.view(R, D), H, D, _tab(kr), _tab(vr), None, seg, tiles, small, S, 0, 0, 1.0, chunks, splits,
                 po, pm, pl, out, fmt)
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
    H, S = 2, 3072
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
    from tensorfold.cuda.sm70.gemma4_native import extension
    ext = extension()
    dev = torch.device("cuda")
    x = torch.zeros(1, 4 * 256, device=dev)
    w = torch.ones(256, device=dev).bfloat16()
    inv = torch.zeros(128, device=dev)
    one = torch.zeros(1, dtype=torch.int32, device=dev)
    ring = torch.zeros(2048, 256, dtype=torch.float16, device=dev)
    p = torch.zeros(2048, dtype=torch.int32, device=dev)
    q = torch.empty(1, 2, 256, dtype=torch.float16, device=dev)
    with pytest.raises(RuntimeError, match="linear 512-wide"):
        ext.qkv_prep(x, 1, 2, 256, 1, w, w, inv, one, one, q, _tab(ring), _tab(ring), _tab(p), 2048, 1e-6, 8)

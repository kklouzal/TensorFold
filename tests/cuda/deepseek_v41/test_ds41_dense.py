"""dmv13 dense projections (tensorfold.cuda.sm70.ds41_dense): plane packing, restore, small-M and GEMM bands,
the vocabulary-shard Q5_K lookup. Synthetic GGUF rows; the real-checkpoint checks skip without the model."""

from pathlib import Path

import numpy as np
import pytest
import torch

cuda = pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0),
                          reason="SM70 CUDA required")
pytestmark = cuda

MODEL = Path("/home/kklouzal/models/ds41-tf/config.json")


def _q8_rows(rng, n, k):
    raw = np.zeros((n, k // 32, 34), np.uint8)
    d = (rng.random((n, k // 32)) * 0.02 + 1e-4).astype(np.float16)
    raw[..., :2] = d.view(np.uint8).reshape(n, k // 32, 2)
    raw[..., 2:] = rng.integers(-128, 128, (n, k // 32, 32)).astype(np.int8).view(np.uint8)
    return raw.reshape(n, -1)


def _q5k_rows(rng, n, k):
    raw = rng.integers(0, 256, (n, k // 256, 176)).astype(np.uint8)
    d = (rng.random((n, k // 256)) * 0.002 + 1e-5).astype(np.float16)
    dmin = (rng.random((n, k // 256)) * 0.004 + 1e-5).astype(np.float16)
    raw[..., 0:2] = d.view(np.uint8).reshape(n, -1, 2)
    raw[..., 2:4] = dmin.view(np.uint8).reshape(n, -1, 2)
    return raw.reshape(n, -1)


def _q5k_fields(raw, k):
    n = raw.shape[0]
    b = raw.reshape(n, k // 256, 176)
    d = b[..., 0:2].copy().view(np.float16)[..., 0].astype(np.float32)
    dmin = b[..., 2:4].copy().view(np.float16)[..., 0].astype(np.float32)
    s = b[..., 4:16].astype(np.int32)
    sc = np.empty(s.shape[:-1] + (8,), np.int32)
    mn = np.empty_like(sc)
    for j in range(4):
        sc[..., j], mn[..., j] = s[..., j] & 63, s[..., j + 4] & 63
    for j in range(4, 8):
        sc[..., j] = (s[..., j + 4] & 0xF) | ((s[..., j - 4] >> 6) << 4)
        mn[..., j] = (s[..., j + 4] >> 4) | ((s[..., j] >> 6) << 4)
    qh, qs = b[..., 16:48].astype(np.int32), b[..., 48:176].astype(np.int32)
    q = np.empty((n, k // 256, 256), np.int32)
    for i in range(4):
        c = qs[..., 32 * i:32 * i + 32]
        q[..., 64 * i:64 * i + 32] = (c & 0xF) | (((qh >> (2 * i)) & 1) << 4)
        q[..., 64 * i + 32:64 * i + 64] = (c >> 4) | (((qh >> (2 * i + 1)) & 1) << 4)
    return d, dmin, sc, mn, q


def dmv13_weights(raw, gtype, k):
    """The FP16 weights 1Cat's dmv13 multiplies (gguf_dmv13_dense.reconstruct; FMA evaluated exactly)."""
    n = raw.shape[0]
    if gtype == 8:
        b = raw.reshape(n, k // 32, 34)
        d = b[..., :2].copy().view(np.float16).astype(np.float64)
        q = b[..., 2:].copy().view(np.int8).astype(np.float64)
        return (q * d).astype(np.float16).reshape(n, k)
    d, dmin, sc, mn, q = _q5k_fields(raw, k)
    s16 = (d[..., None] * sc).astype(np.float32).astype(np.float16).astype(np.float64)
    m16 = (-(dmin[..., None] * mn)).astype(np.float32).astype(np.float16).astype(np.float64)
    w = q.reshape(n, k // 256, 8, 32) * s16[..., None] + m16[..., None]
    return w.astype(np.float16).reshape(n, k)


def ggml_fp32(raw, gtype, k):
    """ggml-quants.c FP32 dequantization (the exact model value of each weight)."""
    n = raw.shape[0]
    if gtype == 8:
        b = raw.reshape(n, k // 32, 34)
        d = b[..., :2].copy().view(np.float16).astype(np.float32)
        return (b[..., 2:].copy().view(np.int8).astype(np.float32) * d).reshape(n, k)
    d, dmin, sc, mn, q = _q5k_fields(raw, k)
    d1 = (d[..., None] * sc).astype(np.float32)
    m1 = (dmin[..., None] * mn).astype(np.float32)
    return (d1[..., None] * q.reshape(n, k // 256, 8, 32).astype(np.float32) - m1[..., None]).reshape(n, k)


def _planes(raw, gtype, k, first_step=0):
    from tensorfold.cuda.sm70 import ds41_dense as dense
    planes = dense.Planes(dense.FORMAT[gtype], raw.shape[0], k, "cuda")
    planes.pack(torch.from_numpy(raw).cuda(), gtype, first_step)
    return planes


@pytest.mark.parametrize("gtype,n,k,first", [(8, 100, 5120, 0), (8, 64, 1024, 32), (13, 96, 1280, 0),
                                              (13, 40, 288, 27), (13, 64, 4096, 0)])
def test_pack_restore_exact(gtype, n, k, first):
    from tensorfold.cuda.sm70 import ds41_dense as dense
    from tensorfold.cuda.sm70 import onecat_ds41 as native
    rng = np.random.default_rng(gtype * 1000 + n + k)
    total = (first * 32 + k + 255) // 256 * 256
    raw = (_q8_rows if gtype == 8 else _q5k_rows)(rng, n, total)
    planes = _planes(raw, gtype, k, first)
    got = planes.restore(torch.empty(n, k, dtype=torch.float16, device="cuda")).cpu().numpy()
    # the restore is exactly the ggml FP16 decode (Q5_K: FP32 coefficients from the kept super-block headers)
    expect = ggml_fp32(raw, gtype, total).astype(np.float16)[:, first * 32:first * 32 + k]
    assert np.array_equal(got.view(np.uint16), expect.view(np.uint16))
    ref = native.dequantize(torch.from_numpy(raw).cuda(), gtype, total)[:, first * 32:first * 32 + k]
    assert torch.equal(ref.cpu(), torch.from_numpy(got))
    # partial restore by tiles
    if n > 32:
        part = planes.restore(torch.empty(n - 32, k, dtype=torch.float16, device="cuda"), tile0=1)
        assert np.array_equal(part.cpu().numpy().view(np.uint16), expect[32:].view(np.uint16))
    blocks = 0 if gtype == 8 else (first + k // 32 + 7) // 8 - first // 8
    assert planes.nbytes == sum(dense.plane_bytes(planes.fmt, n, k)) + n * 16 * blocks


def _group(specs, seed):
    from tensorfold.cuda.sm70 import ds41_dense as dense
    rng = np.random.default_rng(seed)
    scratch = dense.Scratch("cuda", restore_bytes=8 << 20)
    members, weights, exact = [], [], []
    for gtype, n, k in specs:
        total = (k + 255) // 256 * 256
        raw = (_q8_rows if gtype == 8 else _q5k_rows)(rng, n, total)
        members.append(_planes(raw, gtype, k))
        weights.append(torch.from_numpy(dmv13_weights(raw, gtype, total)[:, :k]).cuda())
        exact.append(torch.from_numpy(ggml_fp32(raw, gtype, total).astype(np.float16)[:, :k]).cuda())
    return dense.Group(members, scratch), weights, exact


@pytest.mark.parametrize("specs", [[(8, 1280, 5120), (8, 512, 5120)], [(13, 4096, 1280)], [(13, 5120, 288)],
                                   [(13, 288, 5120), (13, 288, 5120)], [(13, 1024, 4096)], [(13, 512, 5120), (8, 96, 5120)]])
def test_rows_band(specs):
    group, weights, _ = _group(specs, len(specs) + specs[0][1])
    k = specs[0][2]
    torch.manual_seed(0)
    x32 = torch.randn(64, k, dtype=torch.float16, device="cuda")
    full = group(x32)
    for m in (1, 3, 8, 9, 16, 17, 24, 32, 33, 48, 64):
        outs = group(x32[:m].clone())
        for got, ref_all, w in zip(outs, full, weights):
            # every row equals its row in any other launch (1Cat dmv13 rows are batch invariant; patch 112)
            assert torch.equal(got, ref_all[:m])
            ref = x32[:m].double() @ w.double().t()
            err = (got.double() - ref).abs()
            bound = ref.abs() * 2 ** -10 + k * 2 ** -22 * (x32[:m].double().abs() @ w.double().abs().t())
            assert bool((err <= bound + 1e-6).all()), float(err.max())


def test_rows_k_bound_ignores_columns_past_k():
    group, _, _ = _group([(13, 5120, 288)], 7)
    wide = torch.full((8, 384), float("nan"), dtype=torch.float16, device="cuda")
    wide[:, :288] = torch.randn(8, 288, dtype=torch.float16, device="cuda")
    got = group(wide[:, :288])[0]
    assert bool(torch.isfinite(got).all())
    assert torch.equal(got, group(wide[:, :288].contiguous())[0])


def test_gemm_band_matches_restored_weights():
    group, _, weights = _group([(13, 3200, 6144)], 11)    # 3200 x 6144 > 8 MiB restore chunk: two chunks
    x = torch.randn(80, 6144, dtype=torch.float16, device="cuda")
    got = group(x)[0]
    ref = torch.matmul(x, weights[0].t())        # cuBLAS may pick other kernels per chunk width: tolerance
    exact = x.double() @ weights[0].double().t()
    assert float((got.double() - exact).abs().max()) <= 1.5 * float((ref.double() - exact).abs().max()) + 1e-3


def test_q5k_rows_vocab_shard():
    from tensorfold.cuda.sm70 import ds41_dense as dense
    from tensorfold.cuda.sm70 import onecat_ds41 as native
    rng = np.random.default_rng(5)
    table = torch.from_numpy(_q5k_rows(rng, 50, 5120)).cuda()
    ids = torch.tensor([100, 120, 149, 99, 150, 0, 125], dtype=torch.int64, device="cuda")
    out = dense.q5k_rows(table, ids, 100, torch.empty(7, 5120, dtype=torch.float16, device="cuda"))
    ref = native.dequantize(table, 13, 5120)
    for i, token in enumerate(ids.tolist()):
        if 100 <= token < 150:
            assert torch.equal(out[i], ref[token - 100])
        else:
            assert bool((out[i] == 0).all()) and not bool(torch.signbit(out[i]).any())


@pytest.mark.skipif(not MODEL.exists(), reason="DeepSeek-V4.1 lane model absent")
def test_real_projections_against_fp16_decode():
    import json
    from tensorfold.families.deepseek_v4.cuda.weights import Loader, Shards
    config = json.loads(MODEL.read_text())
    shards = Shards(Path(config["gguf_file"]), Path(config["engram_file"]))
    loader = Loader(shards, 3, torch.device("cuda"))
    for name, kind in (("blk.0.attn_q_a.weight", None), ("blk.0.attn_kv.weight", None),
                       ("blk.0.ffn_down_shexp.weight", "row"), ("blk.0.attn_q_b.weight", "column"),
                       ("blk.3.attn_output_b.weight", "row"), ("blk.3.ffn_up_shexp.weight", "column")):
        planes = (loader.planes(name) if kind is None else loader.row_planes(name) if kind == "row"
                  else loader.column_planes(name))
        fp16 = loader.dense(name) if kind is None else loader.row(name) if kind == "row" else loader.column(name)
        got = planes.restore(torch.empty(planes.n, planes.k, dtype=torch.float16, device="cuda"))
        assert torch.equal(got, fp16)   # the prefill weights are the ggml FP16 decode (Q5_K: exact coefficients)


def test_router_gate_bf16_matches_fp32():
    from tensorfold.families.deepseek_v4.cuda.kernels.mhc import fp32_projection
    torch.manual_seed(3)
    w32 = torch.randn(384, 5120, device="cuda").to(torch.bfloat16).float()   # BF16-exact FP32 coefficients
    w16 = w32.to(torch.bfloat16)
    for m in (1, 2, 5, 6, 8, 64):
        x = torch.randn(m, 5120, dtype=torch.float16, device="cuda")
        assert torch.equal(fp32_projection(x, w16), fp32_projection(x, w32))

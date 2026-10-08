"""Packed RotorQuant CUDA numerics, attention, stream and engine lifecycle.

Independent FP64 matrix/byte-loop references live in rotorquant_ref. Exact
encoder comparisons use its staged FP32 reference only where that is the
declared contract; attention uses separate Torch FP64 softmax/SV arithmetic.
"""

import math

import pytest
import torch
import triton
import triton.language as tl

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from tensorfold.families.qwen4_exp.cuda import attention as attn_mod  # noqa: E402
from tensorfold.families.qwen4_exp.cuda import rotorquant_kernel as kernel  # noqa: E402
from tensorfold.families.qwen4_exp.cuda import rotorquant_ref as ref  # noqa: E402
from tensorfold.families.qwen4_exp.kv_formats import FORMATS  # noqa: E402

VARIANTS = ("planar", "isofast")
WIDTHS = (3, 4)
FORMAT_CASES = tuple(item for item in FORMATS if item.rotor is not None)


@triton.jit
def _encode(X, CODE, RMS, N: tl.constexpr, BITS: tl.constexpr, CODEC: tl.constexpr, M: tl.constexpr):
    row = tl.program_id(0) * M + tl.arange(0, M)
    x = tl.load(X + row[:, None] * 128 + tl.arange(0, 128)[None, :], row[:, None] < N, other=0.0)
    if BITS == 3:
        low, high, rms = kernel.quant_groups_3(x, M, CODEC)
        tl.store(CODE + row[:, None] * 48 + tl.arange(0, 32)[None, :], low, row[:, None] < N)
        tl.store(CODE + row[:, None] * 48 + 32 + tl.arange(0, 16)[None, :], high, row[:, None] < N)
    elif BITS == 6:
        low, high, rms = kernel.quant_groups_6(x, M, CODEC)
        tl.store(CODE + row[:, None] * 96 + tl.arange(0, 64)[None, :], low, row[:, None] < N)
        tl.store(CODE + row[:, None] * 96 + 64 + tl.arange(0, 32)[None, :], high, row[:, None] < N)
    elif BITS == 7:
        low, mid, high, rms = kernel.quant_groups_7(x, M, CODEC)
        tl.store(CODE + row[:, None] * 112 + tl.arange(0, 64)[None, :], low, row[:, None] < N)
        tl.store(CODE + row[:, None] * 112 + 64 + tl.arange(0, 32)[None, :], mid, row[:, None] < N)
        tl.store(CODE + row[:, None] * 112 + 96 + tl.arange(0, 16)[None, :], high, row[:, None] < N)
    elif BITS == 8:
        code, rms = kernel.quant_groups_8(x, M, CODEC)
        tl.store(CODE + row[:, None] * 128 + tl.arange(0, 128)[None, :], code, row[:, None] < N)
    else:
        code, rms = kernel.quant_groups_4(x, M, CODEC)
        tl.store(CODE + row[:, None] * 64 + tl.arange(0, 64)[None, :], code, row[:, None] < N)
    tl.store(RMS + row, rms, row < N)


@triton.jit
def _decode(CODE, RMS, OUT, N: tl.constexpr, W: tl.constexpr, BITS: tl.constexpr, M: tl.constexpr):
    row = tl.program_id(0) * M + tl.arange(0, M)
    group = tl.arange(0, W // 128)
    rms = tl.load(RMS + row[:, None] * (W // 128) + group[None, :], row[:, None] < N, other=0.0)
    if BITS == 3:
        il, ih = tl.arange(0, W // 4), tl.arange(0, W // 8)
        low = tl.load(CODE + row[:, None] * (W * 3 // 8) +
                      (il // 32 * 48 + il % 32)[None, :], row[:, None] < N, other=0)
        high = tl.load(CODE + row[:, None] * (W * 3 // 8) +
                       (ih // 16 * 48 + 32 + ih % 16)[None, :], row[:, None] < N, other=0)
        got = kernel.dequant_group_3(low, high, rms, M, W)
    elif BITS == 6:
        il, ih = tl.arange(0, W // 2), tl.arange(0, W // 4)
        low = tl.load(CODE + row[:, None] * (W * 6 // 8) +
                      (il // 64 * 96 + il % 64)[None, :], row[:, None] < N, other=0)
        high = tl.load(CODE + row[:, None] * (W * 6 // 8) +
                       (ih // 32 * 96 + 64 + ih % 32)[None, :], row[:, None] < N, other=0)
        got = kernel.dequant_group_6(low, high, rms, M, W)
    elif BITS == 7:
        il, im, ih = tl.arange(0, W // 2), tl.arange(0, W // 4), tl.arange(0, W // 8)
        low = tl.load(CODE + row[:, None] * (W * 7 // 8) +
                      (il // 64 * 112 + il % 64)[None, :], row[:, None] < N, other=0)
        mid = tl.load(CODE + row[:, None] * (W * 7 // 8) +
                      (im // 32 * 112 + 64 + im % 32)[None, :], row[:, None] < N, other=0)
        high = tl.load(CODE + row[:, None] * (W * 7 // 8) +
                       (ih // 16 * 112 + 96 + ih % 16)[None, :], row[:, None] < N, other=0)
        got = kernel.dequant_group_7(low, mid, high, rms, M, W)
    elif BITS == 8:
        i = tl.arange(0, W)
        code = tl.load(CODE + row[:, None] * W + i[None, :], row[:, None] < N, other=0)
        got = kernel.dequant_group_8(code, rms, M, W)
    else:
        i = tl.arange(0, W // 2)
        code = tl.load(CODE + row[:, None] * (W // 2) + i[None, :], row[:, None] < N, other=0)
        got = kernel.dequant_group_4(code, rms, M, W)
    tl.store(OUT + row[:, None] * W + tl.arange(0, W)[None, :], got, row[:, None] < N)


@triton.jit
def _rotation(X, OUT, N: tl.constexpr, W: tl.constexpr, CODEC: tl.constexpr, INVERSE: tl.constexpr,
              M: tl.constexpr):
    row = tl.program_id(0) * M + tl.arange(0, M)
    x = tl.load(X + row[:, None] * W + tl.arange(0, W)[None, :], row[:, None] < N, other=0.0)
    got = kernel.rotate(x, M, W, CODEC, INVERSE)
    tl.store(OUT + row[:, None] * W + tl.arange(0, W)[None, :], got, row[:, None] < N)


@triton.jit
def _midpoints(X, OUT, N: tl.constexpr, BITS: tl.constexpr, W: tl.constexpr):
    i = tl.arange(0, W)
    x = tl.load(X + i, i < N, other=0.0)
    q = kernel._indices(x, BITS)
    tl.store(OUT + i, q, i < N)


@triton.jit
def _centroid_bits(CODE, OUT, BITS: tl.constexpr):
    row = tl.arange(0, 2)
    coordinate = tl.arange(0, 128)
    offset = row[:, None] * 128 + coordinate[None, :]
    q = tl.load(CODE + offset).to(tl.int32)
    # Observe the production lookup before scale multiplication/BF16 rounding.
    tl.store(OUT + offset, kernel._centroids(q, BITS))


@pytest.mark.parametrize("bits", (7, 8), ids=("rotorquant7", "rotorquant8"))
def test_high_precision_every_centroid_has_exact_canonical_fp32_bits(bits):
    if bits == 8:
        indices = torch.arange(256, dtype=torch.int64).reshape(2, 128)
    else:
        first = torch.arange(128, dtype=torch.int64)
        indices = torch.stack((first, first.flip(0)))
    code = indices.to(dtype=torch.uint8, device="cuda")
    got = torch.empty((2, 128), dtype=torch.float32, device="cuda")
    _centroid_bits[(1,)](code, got, bits, num_warps=4)
    canonical = torch.tensor(ref.centroids(bits), dtype=torch.float32)[indices]
    assert torch.equal(got.cpu().view(torch.int32), canonical.view(torch.int32))


def encode(x, bits, variant, m=4):
    lead, width = x.shape[:-1], x.shape[-1]
    code = torch.empty((*lead, width * bits // 8), dtype=torch.uint8, device=x.device)
    rms = torch.empty((*lead, width // 128), dtype=torch.float32, device=x.device)
    n = x.numel() // 128
    _encode[(triton.cdiv(n, m),)](x, code, rms, n, bits, ref.variant_id(variant), m,
                                enable_fp_fusion=False, num_warps=4)
    return code, rms


def decode(code, rms, bits, width, m=4):
    if code.shape[-1] * 8 != width * bits or rms.shape != (*code.shape[:-1], width // 128):
        raise ValueError("test decoder requires matching payload, group metadata and output width")
    out = torch.empty((*code.shape[:-1], width), dtype=torch.bfloat16, device=code.device)
    n = out.numel() // width
    _decode[(triton.cdiv(n, m),)](code, rms, out, n, width, bits, m,
                                enable_fp_fusion=False, num_warps=4)
    return out


def rotate(x, variant, inverse=False, m=4):
    out = torch.empty_like(x, dtype=torch.float32)
    width, n = x.shape[-1], x.numel() // x.shape[-1]
    _rotation[(triton.cdiv(n, m),)](x, out, n, width, ref.variant_id(variant), inverse, m,
                                  enable_fp_fusion=False, num_warps=4)
    return out


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("rows,width", [(1, 128), (3, 256), (17, 512)])
def test_rotation_matches_dense_matrix_inverse_and_qk_identity(variant, rows, width):
    generator = torch.Generator().manual_seed(111 + rows)
    q = torch.randn((rows, width), generator=generator).float().cuda()
    k = torch.randn((rows, width), generator=generator).float().cuda()
    got = rotate(q, variant)
    independent = torch.tensor([ref.rotate_oracle(row.tolist(), variant) for row in q.cpu().double()],
                               dtype=torch.float64)
    assert torch.allclose(got.cpu().double(), independent, atol=8e-7, rtol=2e-7)
    assert torch.allclose(rotate(got, variant, True), q, atol=2e-6, rtol=4e-7)
    before = (q.double() * k.double()).sum(-1)
    after = (got.double() * rotate(k, variant).double()).sum(-1)
    assert torch.allclose(after, before, atol=2e-5, rtol=2e-6)


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("bits", WIDTHS)
@pytest.mark.parametrize("rows,width", [(1, 128), (3, 256), (17, 512)])
def test_encoded_bytes_and_decoded_tiles_match_declared_reference(variant, bits, rows, width):
    generator = torch.Generator().manual_seed(34 + rows)
    x = torch.randn((rows, 2, width), generator=generator).to(torch.bfloat16).cuda()
    code, rms = encode(x, bits, variant)
    expected_code, expected_rms = ref.quantize_ref(x, bits, variant)
    # Seed37 contains an exact Iso cancellation at a shared zero midpoint.
    # Sum-tree normalization roundings can move that computed coordinate to
    # either side. Bound each index independently; exact parity remains required
    # wherever all legal FP32 trees provably lie in one centroid bin.
    source_rows = x.cpu().reshape(-1, width)
    code_rows = code.cpu().reshape(-1, width * bits // 8)
    expected_rows = expected_code.cpu().reshape(-1, width * bits // 8)
    scale_rows = rms.cpu().reshape(-1, width // 128)
    for source, packed, reference, scales in zip(source_rows, code_rows, expected_rows, scale_rows):
        values = source.float().tolist()
        low, high = ref.index_envelope_oracle(values, bits, variant)
        conditioned_low, conditioned_high = ref.index_envelope_oracle(values, bits, variant, rms=scales.tolist())
        actual = ref.unpack_indices_ref(bytes(packed.tolist()), bits)
        expected = ref.unpack_indices_ref(bytes(reference.tolist()), bits)
        for index, (got, want) in enumerate(zip(actual, expected)):
            assert high[index] - low[index] <= 1
            assert conditioned_low[index] <= got <= conditioned_high[index], (variant, bits, rows, width, index)
            assert low[index] <= want <= high[index]
            if low[index] == high[index]:
                assert got == want == low[index]
            elif got != want:
                assert abs(got - want) == 1
    assert torch.allclose(rms, expected_rms, rtol=3e-7, atol=1e-7)
    back = decode(code, rms, bits, width)
    assert torch.equal(back, ref.dequant_ref(code, rms, bits, variant).to(torch.bfloat16))
    assert code.numel() + rms.nbytes == rows * 2 * (width // 128) * (128 * bits // 8 + 4)
    # Independent byte-loop oracle proves planes/nibbles across group boundaries.
    for packed in code.cpu().reshape(-1, code.shape[-1]):
        payload = bytes(packed.tolist())
        unpacked = ref.unpack_indices_ref(payload, bits)
        assert ref.pack_indices_ref(unpacked, bits) == payload


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("bits", WIDTHS)
def test_certified_margin_sources_require_exact_whole_packed_payload(variant, bits):
    # Dyadic components and row scales are fixed inputs, not a seed search for
    # a passing fixture. The independent global bound certifies every bin first.
    pattern = torch.tensor([.25, -.5, 1.0, -2.0], dtype=torch.bfloat16).repeat(128)
    source = torch.stack([pattern * (2 ** exponent) for exponent in (-3, 0, 3)])[:, None, :].repeat(1, 2, 1)
    for row in source.reshape(-1, 512):
        low, high = ref.index_envelope_oracle(row.float().tolist(), bits, variant)
        assert low == high
    code, rms = encode(source.cuda(), bits, variant)
    expected_code, expected_rms = ref.quantize_ref(source.cuda(), bits, variant)
    assert torch.equal(code, expected_code)
    assert torch.allclose(rms, expected_rms, rtol=3e-7, atol=1e-7)


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("bits", WIDTHS)
@pytest.mark.parametrize("magnitude", [0.0, 10000.0, 1e30])
def test_zero_and_large_finite_bf16_sources_do_not_use_fp16_norms(variant, bits, magnitude):
    source = torch.full((3, 256), magnitude, dtype=torch.bfloat16, device="cuda")
    code, rms = encode(source, bits, variant)
    assert bool(torch.isfinite(rms).all())
    assert rms.dtype == torch.float32
    if magnitude == 0:
        assert torch.equal(rms, torch.zeros_like(rms))
        assert bool((ref.unpack_ref(code, bits) == ((1 << bits) // 2 - 1)).all())
        assert torch.equal(decode(code, rms, bits, 256), torch.zeros_like(source))
    else:
        assert torch.allclose(rms, source[:, :2].float(), rtol=3e-7, atol=0)
        assert bool(torch.isfinite(decode(code, rms, bits, 256)).all())


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("bits", WIDTHS)
def test_bf16_subnormal_and_mixed_zero_groups_preserve_nonzero_rms(variant, bits):
    # Construct and round on CPU, then transfer the actual BF16 bit patterns;
    # GPU source creation must not silently flush the value before the codec.
    tiny = torch.tensor(1e-40).to(torch.bfloat16)
    source = torch.full((3, 256), tiny.item(), dtype=torch.bfloat16)
    source[1, 1::2] = 0
    source[1, 128:] = 0
    source[2, 1::2] = -tiny
    assert tiny.item() != 0
    code, rms = encode(source.cuda(), bits, variant)
    independent_rms = []
    for row in source.double():
        independent_rms.append([math.hypot(*row[start:start + 128].tolist()) / math.sqrt(128)
                                for start in (0, 128)])
    expected = torch.tensor(independent_rms, dtype=torch.float64)
    metadata = rms.cpu().double()
    assert bool((metadata[expected != 0] > 0).all())
    assert bool((metadata[expected == 0] == 0).all())
    # Four FP32 subnormal units cover the staged normalization/reduction rounding
    # while remaining far below a BF16 unit. This must not permit FTZ-to-zero.
    assert torch.allclose(metadata, expected, atol=4 * 2 ** -149, rtol=0)
    restored = decode(code, rms, bits, 256).cpu()
    independent = []
    for packed, norms in zip(code.cpu(), metadata):
        independent.append(ref.dequantize_oracle(bytes(packed.tolist()), norms.tolist(), bits, variant))
    expected_tiles = torch.tensor(independent, dtype=torch.float32).to(torch.bfloat16)
    assert bool(torch.isfinite(restored).all())
    assert torch.equal(restored, expected_tiles)
    assert bool((restored[0] != 0).any())


@pytest.mark.parametrize("bits", WIDTHS)
def test_exact_midpoints_and_fp32_neighbors_use_lower_tie(bits):
    midpoint = torch.tensor(ref.thresholds(bits), dtype=torch.float32, device="cuda")
    below = torch.nextafter(midpoint, torch.full_like(midpoint, -math.inf))
    above = torch.nextafter(midpoint, torch.full_like(midpoint, math.inf))
    source = torch.stack((below, midpoint, above), -1).flatten()
    got = torch.empty_like(source, dtype=torch.int32)
    _midpoints[(1,)](source, got, source.numel(), bits, triton.next_power_of_2(source.numel()))
    expected = torch.stack((torch.arange(len(midpoint)), torch.arange(len(midpoint)),
                            torch.arange(1, len(midpoint) + 1)), -1).flatten().cuda()
    assert torch.equal(got, expected)


@pytest.mark.parametrize("bits", WIDTHS)
def test_every_code_and_plane_boundary_decodes_independent_byte_patterns(bits):
    codes = [(13 * i + i // 127) % (1 << bits) for i in range(512)]
    payload = ref.pack_indices_ref(codes, bits)
    packed = torch.tensor(list(payload), dtype=torch.uint8, device="cuda").reshape(2, 256 * bits // 8)
    rms = torch.tensor([[0.0, 0.25], [1.0, 10000.0]], dtype=torch.float32, device="cuda")
    got = decode(packed, rms, bits, 256).cpu()
    independent = torch.tensor(ref.dequantize_oracle(payload, rms.cpu().flatten().tolist(), bits),
                               dtype=torch.float32).reshape(2, 256).to(torch.bfloat16)
    assert torch.equal(got, independent)


def dense_attention_oracle(q_rotated, k_rotated, v_rotated, pos, rows, variant):
    """Different implementation: complete causal FP64 softmax/SV, then matrix inverse."""
    q, k, v = (x.cpu().double() for x in (q_rotated, k_rotated, v_rotated))
    heads, hk, width = q.shape[1], k.shape[1], q.shape[-1]
    result = []
    for row in range(rows):
        end = pos + row + 1
        out = []
        for head in range(heads):
            kh = head // (heads // hk)
            score = (k[:end, kh] @ q[row, head]) / math.sqrt(width)
            probability = torch.softmax(score, dim=0)
            value = probability @ v[:end, kh]
            out.append(ref.rotate_oracle(value.tolist(), variant, inverse=True))
        result.append(out)
    return torch.tensor(result, dtype=torch.float64)


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("bits", WIDTHS)
@pytest.mark.parametrize("g,pos,rows,capacity", [(4, 20, 4, 64), (12, 504, 16, 640)])
def test_causal_gqa_attention_matches_independent_full_softmax_value_sum(variant, bits, g, pos, rows, capacity):
    generator = torch.Generator().manual_seed(98 + g)
    hk, width = 2, 256
    q = (torch.randn((rows, hk * g, width), generator=generator) * .4).to(torch.bfloat16).cuda()
    k = (torch.randn((capacity, hk, width), generator=generator) * .5).to(torch.bfloat16).cuda()
    v = (torch.randn((capacity, hk, width), generator=generator) * .5).to(torch.bfloat16).cuda()
    kp, ks = encode(k, bits, variant)
    vp, vs = encode(v, bits, variant)
    qr = rotate(q, variant).to(torch.bfloat16)
    pos0 = torch.tensor([pos], dtype=torch.int32, device="cuda")
    scratch = attn_mod.AttnScratch(rows, hk * g, width, capacity, "cuda", budget=2048, ratio=4)
    got = attn_mod.attention(qr, kp, vp, pos0, scratch, rows, width ** -.5, ks=ks, vs=vs,
                             bits=bits, codec=ref.variant_id(variant)).cpu().double()
    expected = dense_attention_oracle(qr, decode(kp, ks, bits, width), decode(vp, vs, bits, width),
                                     pos, rows, variant)
    assert torch.allclose(got, expected, atol=.003, rtol=.03), (variant, bits, float((got - expected).abs().max()))


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("bits", WIDTHS)
def test_nondefault_stream_encoding_observes_its_producer_and_releases_after_use(variant, bits):
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        source = torch.arange(3 * 256, device="cuda", dtype=torch.float32).reshape(3, 256) / 51
        source = source.to(torch.bfloat16)
        code, rms = encode(source, bits, variant)
        reconstructed = decode(code, rms, bits, 256)
        completion = torch.cuda.Event()
        completion.record()
    torch.cuda.current_stream().wait_event(completion)
    expected_code, expected_rms = ref.quantize_ref(source, bits, variant)
    assert torch.equal(code, expected_code)
    assert torch.allclose(rms, expected_rms, rtol=3e-7, atol=1e-7)
    assert torch.equal(reconstructed, ref.dequant_ref(code, rms, bits, variant).to(torch.bfloat16))


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("bits", WIDTHS)
def test_codec_graph_replays_changed_inputs_without_hidden_norm_or_pointer_state(variant, bits):
    source = torch.randn((3, 256), device="cuda", dtype=torch.bfloat16)
    # Precompile and satisfy graph-capture dependency on a warmup stream.
    warm = torch.cuda.Stream()
    warm.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warm):
        warm_code, warm_rms = encode(source, bits, variant)
        decode(warm_code, warm_rms, bits, 256)
    torch.cuda.current_stream().wait_stream(warm)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        code, rms = encode(source, bits, variant)
        restored = decode(code, rms, bits, 256)
    for magnitude in (0.0, 7.0, 10000.0):
        source.fill_(magnitude)
        graph.replay()
        expected_code, expected_rms = ref.quantize_ref(source, bits, variant)
        assert torch.equal(code, expected_code)
        assert torch.allclose(rms, expected_rms, rtol=3e-7, atol=1e-7)
        assert torch.equal(restored, ref.dequant_ref(code, rms, bits, variant).to(torch.bfloat16))


@pytest.mark.parametrize("format", FORMAT_CASES, ids=lambda item: item.name)
def test_cache_growth_clone_and_accounting_preserve_payload_and_rms(format):
    from tensorfold.families.qwen4_exp.cuda.kvcache import KVCache, row_bytes

    cache = KVCache(17, 2, 256, "cuda", format.name)
    assert cache.format == format and cache.codec == format.codec and cache.bits == format.bits
    assert cache.ks.dtype == torch.float32 and cache.ks.shape == (17, 2, 2)
    assert cache.nbytes == 17 * row_bytes(2, 256, format.name)
    generator = torch.Generator(device="cuda").manual_seed(319)
    source = torch.randn((17, 2, 256), device="cuda", generator=generator, dtype=torch.bfloat16)
    for payload, rms in ((cache.k, cache.ks), (cache.v, cache.vs)):
        code, scale = encode(source, format.bits, format.rotor.variant)
        payload.copy_(code)
        rms.copy_(scale)
    for copied in (cache.clone(), cache.resized(64, 13)):
        count = 17 if copied.capacity == 17 else 13
        assert copied.format == cache.format
        for ours, theirs in ((copied.k, cache.k), (copied.v, cache.v),
                             (copied.ks, cache.ks), (copied.vs, cache.vs)):
            assert torch.equal(ours[:count], theirs[:count])
            assert ours.data_ptr() != theirs.data_ptr()
    with pytest.raises(ValueError, match="128"):
        KVCache(17, 2, 64, "cuda", format.name)


@pytest.mark.parametrize("format", FORMAT_CASES, ids=lambda item: item.name)
@pytest.mark.parametrize("position", [0, 262143, 524284])
def test_fused_prep_quantizes_post_rope_bf16_and_leaves_index_keys_unchanged(format, position):
    from test_flashnext_forward import _model
    from tensorfold.families.qwen4_exp.cuda import glue
    from tensorfold.families.qwen4_exp.cuda.kvcache import KVCache

    w = _model()
    c = w.cfg
    a = next(layer.attn for layer in w.layers if layer.attn is not None)
    rows = 3
    baseline = KVCache(rows, c.kv_heads, c.head_dim, "cuda", "bf16")
    candidate = KVCache(rows, c.kv_heads, c.head_dim, "cuda", format.name)
    # The prep kernel's absolute destination offset requires real allocated
    # backing. The fixture instead uses a rotary delta and position zero:
    # this produces the identical text angle while keeping cache addresses small.
    pos = torch.zeros(1, dtype=torch.int32, device="cuda")
    delta = torch.tensor([position], dtype=torch.int32, device="cuda")
    width = c.heads * 2 * c.head_dim + 2 * c.kv_heads * c.head_dim + (c.index_heads + 1) * c.index_dim
    generator = torch.Generator(device="cuda").manual_seed(993 + position)
    projected = torch.randn((rows, width), generator=generator, device="cuda", dtype=torch.bfloat16)
    outputs = []
    for cache in (baseline, candidate):
        status = torch.tensor([0, -2], dtype=torch.int32, device="cuda") if cache.codec else None
        q = torch.empty((rows, c.heads, c.head_dim), dtype=torch.bfloat16, device="cuda")
        iq = torch.empty((rows, c.index_heads, c.index_dim), dtype=torch.bfloat16, device="cuda")
        index = torch.empty((rows, c.index_dim), dtype=torch.bfloat16, device="cuda")
        glue.attn_prep(projected, pos, a.q_scale, a.k_scale, a.iq_scale, w.inv_freq, q, cache.k, cache.v,
                       iq, index, c.eps, q_heads=c.heads, kv_heads=c.kv_heads, head_dim=c.head_dim,
                       index_heads=c.index_heads, index_dim=c.index_dim, ks=cache.ks, vs=cache.vs,
                       bits=cache.bits if cache.quantized else 0, codec=cache.codec, delta=delta, status=status)
        if status is not None:
            assert status.cpu().tolist() == [0, -2]
        outputs.append((q, iq, index))
    for source, code, scale in ((baseline.k, candidate.k, candidate.ks),
                                (baseline.v, candidate.v, candidate.vs)):
        expected, rms = ref.quantize_ref(source, format.bits, format.rotor.variant)
        if format.codec <= 2:
            assert torch.equal(code, expected)
            assert torch.allclose(scale, rms, rtol=3e-7, atol=1e-7)
        else:
            for row, payload, norm, reference in zip(source.cpu().reshape(-1, c.head_dim).float().tolist(),
                                                     code.cpu().reshape(-1, c.head_dim*format.bits//8).tolist(),
                                                     scale.cpu().reshape(-1, c.head_dim//128).tolist(),
                                                     expected.cpu().reshape(-1, c.head_dim*format.bits//8).tolist()):
                indices = ref.unpack_indices_ref(bytes(payload), format.bits)
                reference = ref.unpack_indices_ref(bytes(reference), format.bits)
                lower,upper = ref.index_envelope_oracle(row, format.bits, format.rotor.variant, rms=norm)
                assert all(low <= a <= high and low <= b <= high and high-low <= 1 for low,a,b,high in
                           zip(lower,indices,reference,upper))
    assert torch.equal(outputs[1][0], rotate(outputs[0][0], format.rotor.variant).to(torch.bfloat16))
    assert torch.equal(outputs[1][1], outputs[0][1])
    assert torch.equal(outputs[1][2], outputs[0][2])


@pytest.mark.parametrize("format", FORMAT_CASES, ids=lambda item: item.name)
def test_engine_windows_and_chunkings_keep_identical_format_semantics(format):
    from test_flashnext_kvquant import test_chunked_prefill_is_one_shot as chunk_contract
    from test_flashnext_kvquant import test_windows_match_serial_steps as window_contract

    window_contract(format.name, format.bits)
    chunk_contract(format.name, format.bits)


@pytest.mark.parametrize("format", FORMAT_CASES, ids=lambda item: item.name)
@pytest.mark.parametrize("sampled", [False, True])
def test_engine_mtp_and_graphs_emit_the_same_keyed_samples_as_serial(format, sampled):
    from test_flashnext_kvquant import test_drafted_equals_serial as draft_contract
    from tensorfold.engine.exact_sampling import Sampling

    sampling = Sampling(seed=1234, top_k=20, top_p=.95) if sampled else None
    draft_contract(format.name, format.bits, sampling)


@pytest.mark.parametrize("format", FORMAT_CASES, ids=lambda item: item.name)
def test_qsa_prompt_blocks_and_prefix_restore_read_only_committed_packed_rows(format):
    from test_flashnext_kvquant import test_sparse_rows_and_prompt_blocks_read_the_quantized_cache as sparse_contract

    sparse_contract(format.name, None)


@pytest.mark.parametrize("format", FORMAT_CASES, ids=lambda item: item.name)
def test_multi_slot_capacity_growth_updates_all_packed_payload_and_scale_addresses(format):
    from test_flashnext_multi import test_streams_that_grow_past_their_first_rows_equal_each_alone as growth_contract

    growth_contract(format.name)


@pytest.mark.parametrize("format", FORMAT_CASES, ids=lambda item: item.name)
def test_yarn2_mrope_mtp_graphs_and_concurrent_image_paths_match_serial(format):
    from test_flashnext_yarn import test_yarn_mtp_graphs_and_concurrent_image_attention_match_serial as combined_contract

    combined_contract(format.name)


@pytest.mark.parametrize("format", FORMAT_CASES, ids=lambda item: item.name)
def test_prefix_copy_keeps_format_identity_and_rejects_legacy_equal_bit_width(format):
    from test_flashnext_forward import _model
    from tensorfold.families.qwen4_exp.cuda.decode import Engine, prefill

    w = _model()
    source = Engine(w, capacity=128, max_rows=8, prefill_rows=16, kv_dtype=format.name)
    target = Engine(w, capacity=128, max_rows=8, prefill_rows=16, kv_dtype=format.name)
    prompt = [5, 17, 99, 250, 1023, 7, 64, 300, 11, 12, 13]
    prefill(source, prompt, None)
    target.st.copy_prefix(source.st, source.st.pos, source.st.mtp_len)
    for original, copied in zip(source.st.kc + [source.st.mtp_kc], target.st.kc + [target.st.mtp_kc]):
        assert original.format == copied.format == format
        n = source.st.mtp_len if original is source.st.mtp_kc else source.st.pos
        for a, b in ((original.k, copied.k), (original.v, copied.v),
                     (original.ks, copied.ks), (original.vs, copied.vs)):
            assert torch.equal(a[:n], b[:n])
    legacy = Engine(w, capacity=128, max_rows=8, prefill_rows=16, kv_dtype="int4")
    with pytest.raises(ValueError, match="matching cache formats"):
        legacy.st.copy_prefix(source.st, source.st.pos, source.st.mtp_len)


def test_actual_loss_harness_all_rows_match_fresh_public_prefill_endpoints():
    """Independent endpoint oracle, preserving the native prefill arithmetic.

    Native decode and prefill have separate arithmetic contracts. Comparing
    this instrumented prefill with fresh public-prefill endpoints isolates the
    head instrumentation without incorrectly imposing decode bit identity.
    """
    from test_flashnext_forward import _model
    from tensorfold.families.qwen4_exp.cuda.decode import Engine, prefill
    from tensorfold.families.qwen4_exp.cuda.forward import compute, stage

    w = _model()
    prompt = ([5, 17, 99, 250, 1023, 7, 64, 300, 11, 12, 13] * 3) + [401, 33]
    expected = []
    for end in range(1, len(prompt) + 1):
        fresh = Engine(w, capacity=128, max_rows=8, prefill_rows=64)
        prefill(fresh, prompt[:end], None, mtp=False)
        expected.append(fresh.pbuf.logits[0].clone())
    batch = Engine(w, capacity=128, max_rows=8, prefill_rows=64)
    # Only this loss harness asks the head for all 35 positions; ordinary
    # serving allocates a bounded number of prompt-ending head rows.
    batch.pbuf.logits = torch.empty((64, w.head.n), dtype=torch.bfloat16, device=w.device)
    got = compute(w, stage(w, batch.pbuf, [(batch.st, prompt)]), batch.pbuf, ends=range(len(prompt))).clone()
    assert got.shape[0] == len(prompt)
    assert torch.equal(got, torch.stack(expected))


@pytest.mark.parametrize("format", FORMAT_CASES, ids=lambda item: item.name)
@pytest.mark.parametrize("failure,bit", [("raw_q_nan", 1), ("raw_k_inf", 2), ("raw_v_nan", 4),
                                          ("raw_v_range", 4), ("post_q_range", 1), ("index_nan", 16)])
def test_fused_numeric_guard_reports_source_and_first_layer_without_publication(format, failure, bit):
    from test_flashnext_forward import _model
    from tensorfold.families.qwen4_exp.cuda import glue
    from tensorfold.families.qwen4_exp.cuda.state import KVNumericError, State

    w = _model()
    c, a = w.cfg, w.layers[1].attn
    st = State(w, 8, 4, format.name)
    width = c.heads * 2 * c.head_dim + 2 * c.kv_heads * c.head_dim + (c.index_heads + 1) * c.index_dim
    projected = torch.full((1, width), .25, dtype=torch.bfloat16, device="cuda")
    # Projection layout is independently defined by Q, gate, K, V and index
    # widths; corrupt a single foreign source, rather than the status oracle.
    q0, k0 = 0, 2 * c.heads * c.head_dim
    v0 = k0 + c.kv_heads * c.head_dim
    i0 = v0 + c.kv_heads * c.head_dim
    if failure == "raw_q_nan":
        projected[0, q0] = float("nan")
    elif failure == "raw_k_inf":
        projected[0, k0] = float("inf")
    elif failure == "raw_v_nan":
        projected[0, v0] = float("nan")
    elif failure == "raw_v_range":
        projected[0, v0] = 2**31
    elif failure == "post_q_range":
        a.q_scale.fill_(2**31)
    else:
        projected[0, i0] = float("nan")
    q = torch.empty((1, c.heads, c.head_dim), dtype=torch.bfloat16, device="cuda")
    iq = torch.empty((1, c.index_heads, c.index_dim), dtype=torch.bfloat16, device="cuda")
    cache = st.kc[0]
    st.kv_begin()
    glue.attn_prep(projected, st.pos_dev, a.q_scale, a.k_scale, a.iq_scale, w.inv_freq,
                   q, cache.k, cache.v, iq, st.ikc[0], c.eps,
                   q_heads=c.heads, kv_heads=c.kv_heads, head_dim=c.head_dim,
                   index_heads=c.index_heads, index_dim=c.index_dim, ks=cache.ks, vs=cache.vs,
                   bits=cache.bits, codec=cache.codec, status=st.kv_status, layer=17)
    with pytest.raises(KVNumericError, match="layer=17"):
        st.kv_check()
    assert int(st.kv_status[0].item()) & bit
    assert st.kv_status[1].item() == 17
    assert st.pos == st.mtp_len == 0
    with pytest.raises(KVNumericError, match="requires reset"):
        st.snapshot()


@pytest.mark.parametrize("format", FORMAT_CASES, ids=lambda item: item.name)
def test_poisoned_state_cannot_clone_grow_copy_restore_or_commit_and_reset_recovers(format):
    from test_flashnext_forward import _model
    from tensorfold.families.qwen4_exp.cuda.decode import Engine, prefill
    from tensorfold.families.qwen4_exp.cuda.forward import _kv_check, commit
    from tensorfold.families.qwen4_exp.cuda.state import KVNumericError

    w = _model()
    e = Engine(w, capacity=64, max_rows=4, prefill_rows=16, kv_dtype=format.name)
    good = e.st.clone()
    snap = good.snapshot()
    pointer = e.st.kv_status.data_ptr()
    e.st.kv_begin()
    e.st.kv_status.copy_(torch.tensor([4, -1], dtype=torch.int32, device="cuda"))
    # Check all participants even when an earlier participant fails, so a later
    # invalid slot cannot survive as an apparently valid prefix.
    good.kv_begin()
    good.kv_status.copy_(torch.tensor([2, 9], dtype=torch.int32, device="cuda"))
    with pytest.raises(KVNumericError, match="layer=MTP"):
        _kv_check([(e.st, 0, 1), (good, 1, 2)])
    assert e.st._kv_error is not None and good._kv_error is not None
    for operation in (e.st.kv_begin, e.st.clone, e.st.snapshot,
                      lambda: e.st.resize(128), lambda: e.st.ensure(65),
                      lambda: e.st.restore(snap), lambda: e.st.copy_prefix(good, 0, 0),
                      lambda: commit(w, e.st, e.buf, 1, 1),
                      lambda: e.sample(torch.zeros((1, w.head.n), device="cuda"), [0], None)):
        with pytest.raises(KVNumericError):
            operation()
    assert e.st.pos == 0 and e.st.capacity == 64
    e.reset()
    assert e.st.kv_status.data_ptr() == pointer
    assert e.st.kv_status.cpu().tolist() == [0, -2]
    assert not e.st._kv_pending and e.st._kv_error is None
    prefill(e, [5, 17, 99, 7], None)
    assert e.st.pos == 4 and e.st.kv_status[0].item() == 0
    snapshot = e.st.snapshot()
    assert snapshot["kv_identity"] == (e.st.kv_pair.identity, "stored-basis-native64-v1")
    assert snapshot["kv_status"] == (0, -2)
    incompatible = dict(snapshot, kv_status=(4, -1))
    before = e.st.rec.clone()
    with pytest.raises(ValueError, match="validated status"):
        e.st.restore(incompatible)
    assert torch.equal(before, e.st.rec)


@pytest.mark.parametrize("format", FORMAT_CASES, ids=lambda item: item.name)
@pytest.mark.parametrize("mtp", [False, True])
def test_graph_numeric_failure_is_checked_after_replay_and_recovers_on_reset(format, mtp):
    from test_flashnext_forward import _model
    from tensorfold.families.qwen4_exp.cuda.decode import Engine, prefill
    from tensorfold.families.qwen4_exp.cuda.state import KVNumericError

    w = _model()
    e = Engine(w, capacity=64, max_rows=4, prefill_rows=16, graphs=True, kv_dtype=format.name)
    prefill(e, [5, 17, 99, 7], None)
    run = (lambda: e.mtp_forward([11], e.last_streams)) if mtp else lambda: e.forward([11])
    run()
    captures = e.graphs.captures
    scale = w.mtp.layer.attn.q_scale if mtp else w.layers[1].attn.q_scale
    saved = scale.clone()
    scale.fill_(float("nan"))
    try:
        with pytest.raises(KVNumericError, match="layer=MTP" if mtp else "layer=1"):
            run()
        assert e.graphs.captures == captures
        with pytest.raises(KVNumericError):
            run()
    finally:
        scale.copy_(saved)
    e.reset()
    prefill(e, [5, 17, 99, 7], None)
    assert torch.isfinite(run()).all()
    assert e.graphs.captures == captures


@pytest.mark.parametrize("format", FORMAT_CASES, ids=lambda item: item.name)
def test_failed_packed_prompt_pass_publishes_nothing_and_reclaimed_slots_remain_usable(format, monkeypatch):
    from test_flashnext_forward import _model
    from tensorfold.cuda.streams import Stream
    from tensorfold.families.qwen4_exp.cuda import glue
    from tensorfold.families.qwen4_exp.cuda.multi import MultiDecoder
    from tensorfold.families.qwen4_exp.cuda.state import KVNumericError

    w = _model()
    dec = MultiDecoder(w, slots=2, capacity=128, depth=1, kv_dtype=format.name, prefill_rows=32)
    emissions = []
    requests = [Stream([5, 17, 99, 7], 1, draft=False, emit=lambda new: emissions.extend(new)),
                Stream([7, 64, 300, 11], 1, draft=False, emit=lambda new: emissions.extend(new))]
    for request in requests:
        dec.admit(request)
    victim = requests[0].st
    original = glue.attn_prep

    def corrupted_projection(projected, *args, **kwargs):
        if kwargs.get("status") is victim.kv_status:
            c = w.cfg
            value_offset = 2 * c.heads * c.head_dim + c.kv_heads * c.head_dim
            projected[0, value_offset] = float("nan")
        return original(projected, *args, **kwargs)

    monkeypatch.setattr(glue, "attn_prep", corrupted_projection)
    finished = dec.round()
    assert len(finished) == 2 and not emissions
    assert all(request.done and isinstance(request.error, KVNumericError) and not request.out
               for request in requests)
    assert victim._kv_error is not None and not dec.kept
    dec.finish(finished)
    assert len(dec.free) == 2 and not dec.live()
    assert all(st.kv_status.cpu().tolist() == [0, -2] for st in dec.free)
    monkeypatch.setattr(glue, "attn_prep", original)
    recovered = Stream([5, 17, 99, 7], 1, draft=False)
    dec.admit(recovered)
    dec.finish(dec.round())
    assert recovered.done and recovered.error is None and len(recovered.out) == 1


@pytest.mark.parametrize("format", FORMAT_CASES, ids=lambda item: item.name)
def test_invalid_kept_prefix_is_rejected_removed_and_reset_before_reuse(format):
    from test_flashnext_forward import _model
    from tensorfold.families.qwen4_exp.cuda.decode import Engine, prefill
    from tensorfold.families.qwen4_exp.cuda.multi import MultiDecoder
    from tensorfold.families.qwen4_exp.cuda.state import KVNumericError

    w = _model()
    dec = MultiDecoder(w, slots=2, capacity=128, depth=1, kv_dtype=format.name, prefill_rows=16)
    st = dec.free.pop()
    e = Engine(w, capacity=128, max_rows=4, prefill_rows=16, kv_dtype=format.name)
    e.st = st
    prompt = [5, 17, 99, 7]
    prefill(e, prompt, None)
    dec._remember(prompt, st, st.snapshot(), None)
    st.kv_begin()
    st.kv_status.copy_(torch.tensor([2, 1], dtype=torch.int32, device="cuda"))
    with pytest.raises(KVNumericError):
        dec._slot_for(prompt + [11], True)
    assert not dec.kept and len(dec.free) == 2
    assert st._kv_error is None and st.kv_status.cpu().tolist() == [0, -2]
    selected, resume, cached = dec._slot_for(prompt + [11], True)
    assert resume is None and cached == 0
    selected.kv_check()


@pytest.mark.parametrize("dtype", ["bf16", "int8", "int4"])
def test_native_states_keep_no_numeric_status_allocation_and_snapshot_identity(dtype):
    from test_flashnext_forward import _model
    from tensorfold.families.qwen4_exp.cuda.state import State

    w = _model()
    st = State(w, 8, 4, dtype)
    assert st.kv_status is None
    st.kv_begin()
    st.kv_check()
    assert not st._kv_pending
    snap = st.snapshot()
    assert snap["kv_identity"] == (st.kv_pair.identity, "stored-basis-native64-v1")
    assert snap["kv_status"] == (0, -2)
    st.restore(snap)


WIDE_VARIANTS = ("iso64-norm", "signed-iso128", "signed-iso64-norm")
NORM_SIX_VARIANTS = ("signed-iso128-norm", "signed-iso128-outlier-norm")
WIDE_BITS = (tuple((variant, 4) for variant in WIDE_VARIANTS) + (("signed-iso128", 6),)
             + tuple((variant, 6) for variant in NORM_SIX_VARIANTS)
             + (("signed-iso128",7),("signed-iso128",8)))


@pytest.fixture(scope="module")
def wide_oracle_module():
    """Load project-owned CPU test oracles without optimized transform reuse."""
    import importlib.util
    from pathlib import Path

    path = Path(__file__).parents[1] / "test_rotorquant_wide.py"
    spec = importlib.util.spec_from_file_location("rotorquant_independent_wide", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("variant", WIDE_VARIANTS + NORM_SIX_VARIANTS)
@pytest.mark.parametrize("rows,width", [(1, 128), (3, 256), (17, 512)])
def test_wide_rotation_matches_independent_matrix_and_true_inverse(variant, rows, width, wide_oracle_module):
    import numpy as np

    codec = ref.variant_id(variant)
    generator = torch.Generator().manual_seed(111 + rows)
    source = torch.randn((rows, width), generator=generator).to(torch.bfloat16).float()
    got = rotate(source.cuda(), variant).cpu().double()
    matrix = wide_oracle_module.wide_matrix(codec)
    expected = source.double().numpy().reshape(-1, 128) @ matrix.T
    # The normalization envelope includes its own positive error terms. Scaling
    # it by the exact raw RMS conservatively bounds rotation-only RN error too.
    bounds = []
    for group in source.reshape(-1, 128):
        # CODEC6/7 share CODEC4's exact rotation. CODEC7's full encoding
        # envelope additionally includes alpha and cannot bound rotation-only
        # output in its original unscaled basis.
        transform_variant = "signed-iso128" if codec in (6,7) else variant
        _, error, rms, _ = ref.rounding_envelope_oracle(group.tolist(), transform_variant)
        raw_rms = math.hypot(*group.tolist()) / math.sqrt(128)
        del rms
        bounds.extend(e * raw_rms for e in error)
    assert np.all(np.abs(got.numpy().reshape(-1) - expected.reshape(-1)) <= np.array(bounds))
    inverse = rotate(got.float().cuda(), variant, inverse=True).cpu().double().numpy().reshape(-1, 128)
    inverse_expected = got.numpy().reshape(-1, 128) @ matrix
    # Twelve signed-sum operations bound three inverse quaternion stages;
    # optional Givens adds two. Absolute sums cover cancellation and FP64 oracle
    # roundoff. This is derived from operation depth, not the observed mismatch.
    steps = 14 if codec in (4,6,7) else 12
    u = 2**-24
    bound = steps*u/(1-steps*u) * (np.abs(got.numpy().reshape(-1, 128)) @ np.abs(matrix)) + 1e-12
    assert np.all(np.abs(inverse - inverse_expected) <= bound)
    restored = source.double().numpy().reshape(-1, 128) @ matrix.T @ matrix
    assert np.all(np.abs(inverse - restored) <= bound + np.array(bounds).reshape(-1, 128) @ np.abs(matrix) + 1e-12)


@pytest.mark.parametrize("variant,bits", WIDE_BITS)
def test_wide_certified_inverse_fixture_requires_exact_packed_payload(variant, bits, wide_oracle_module):
    codec = ref.variant_id(variant)
    original, expected = wide_oracle_module.certified_source(codec, ref.centroids(bits), ref.thresholds(bits))
    group = torch.tensor(original, dtype=torch.bfloat16)
    base = group.repeat(4)
    source = torch.stack([base * (2**power) for power in (-3, 0, 3)])[:, None].repeat(1, 2, 1)
    packed, metadata = encode(source.cuda(), bits, variant)
    payloads = packed.cpu().reshape(-1, 512*bits//8).tolist()
    scales = metadata.cpu().reshape(-1, 4).tolist()
    for row, payload, norm in zip(source.reshape(-1, 512).float().tolist(), payloads, scales):
        indices = ref.unpack_indices_ref(bytes(payload), bits)
        low, high = ref.index_envelope_oracle(row, bits, variant, rms=norm)
        assert low == high == indices == expected * 4
        for offset in range(4):
            begin = offset * 128
            bound = wide_oracle_module.scale_interval(row[begin:begin+128], indices[begin:begin+128],
                                                     ref.centroids(bits), ref.norm_corrected(variant))
            assert bound[0] <= norm[offset] <= bound[1]
    # Reconstruction is independently assembled from actual bytes and metadata,
    # then rounded to FP32/BF16 in the declared order.
    decoded = decode(packed, metadata, bits, 512).cpu()
    independent = torch.tensor([ref.dequantize_oracle(bytes(payload), norm, bits, variant)
                                for payload,norm in zip(payloads,scales)], dtype=torch.float32).to(torch.bfloat16)
    assert torch.equal(decoded.reshape(-1, 512), independent)


@pytest.mark.parametrize("variant,bits", WIDE_BITS)
@pytest.mark.parametrize("rows,width", [(3, 256), (17, 512)])
def test_wide_bf16_cancellation_indices_and_corrected_metadata_obey_analytic_bounds(variant, bits, rows, width, wide_oracle_module):
    codec = ref.variant_id(variant)
    generator = torch.Generator().manual_seed(34 + rows)
    source = torch.randn((rows, 2, width), generator=generator).to(torch.bfloat16)
    packed, metadata = encode(source.cuda(), bits, variant)
    cpu_payload, cpu_metadata = ref.quantize_ref(source, bits, variant)
    for row,payload,norm,reference,reference_norm in zip(source.reshape(-1, width).float().tolist(),
                                                       packed.cpu().reshape(-1, width*bits//8).tolist(),
                                                       metadata.cpu().reshape(-1, width//128).tolist(),
                                                       cpu_payload.reshape(-1, width*bits//8).tolist(),
                                                       cpu_metadata.reshape(-1, width//128).tolist()):
        actual = ref.unpack_indices_ref(bytes(payload), bits)
        other = ref.unpack_indices_ref(bytes(reference), bits)
        low,high = ref.index_envelope_oracle(row, bits, variant, rms=norm)
        for index,(lower,upper,got,want) in enumerate(zip(low,high,actual,other)):
            assert upper-lower <= 1 and lower <= got <= upper and lower <= want <= upper
            if lower == upper:
                assert got == want == lower
            elif got != want:
                assert abs(got-want) == 1, index
        for group in range(width//128):
            begin = group*128
            lower,upper = wide_oracle_module.wide_index_envelope(row[begin:begin+128], codec, ref.thresholds(bits))
            assert lower == low[begin:begin+128] and upper == high[begin:begin+128]
            for indices,scale in ((actual,norm[group]),(other,reference_norm[group])):
                interval = wide_oracle_module.scale_interval(row[begin:begin+128],indices[begin:begin+128],
                                                            ref.centroids(bits),ref.norm_corrected(variant))
                assert interval[0] <= scale <= interval[1]


@pytest.mark.parametrize("variant,bits", WIDE_BITS)
@pytest.mark.parametrize("g,pos,rows,capacity", [(4, 20, 4, 64), (12, 504, 16, 640)])
def test_wide_causal_gqa_attention_matches_independent_matrix_softmax_sv(variant, bits, g, pos, rows, capacity, wide_oracle_module):
    generator = torch.Generator().manual_seed(98 + g)
    hk,width = 2,256
    q = (torch.randn((rows,hk*g,width),generator=generator)*.4).to(torch.bfloat16).cuda()
    k = (torch.randn((capacity,hk,width),generator=generator)*.5).to(torch.bfloat16).cuda()
    v = (torch.randn((capacity,hk,width),generator=generator)*.5).to(torch.bfloat16).cuda()
    kp,ks = encode(k,bits,variant)
    vp,vs = encode(v,bits,variant)
    qr = rotate(q,variant).to(torch.bfloat16)
    scratch = attn_mod.AttnScratch(rows,hk*g,width,capacity,"cuda",budget=2048,ratio=4)
    got = attn_mod.attention(qr,kp,vp,torch.tensor([pos],dtype=torch.int32,device="cuda"),scratch,
                             rows,width**-.5,ks=ks,vs=vs,bits=bits,codec=ref.variant_id(variant)).cpu().double()
    # Decode from independent byte loops, not the kernel being evaluated.
    keys = torch.tensor([ref.dequantize_oracle(bytes(code),scale,bits,variant) for code,scale in
                         zip(kp.cpu().reshape(-1,width*bits//8).tolist(),ks.cpu().reshape(-1,2).tolist())],dtype=torch.float32)
    values = torch.tensor([ref.dequantize_oracle(bytes(code),scale,bits,variant) for code,scale in
                           zip(vp.cpu().reshape(-1,width*bits//8).tolist(),vs.cpu().reshape(-1,2).tolist())],dtype=torch.float32)
    keys = keys.to(torch.bfloat16).double().reshape(capacity,hk,width)
    values = values.to(torch.bfloat16).double().reshape(capacity,hk,width)
    queries = qr.cpu().double()
    matrix = torch.tensor(wide_oracle_module.wide_matrix(ref.variant_id(variant)),dtype=torch.float64)
    expected = torch.empty_like(got)
    for row in range(rows):
        end = pos+row+1
        for head in range(hk*g):
            source_head = head//g
            probability = torch.softmax(keys[:end,source_head] @ queries[row,head] / math.sqrt(width),dim=0)
            rotated_value = probability @ values[:end,source_head]
            expected[row,head] = (rotated_value.reshape(-1,128) @ matrix).reshape(width)
    assert torch.allclose(got,expected,atol=.003,rtol=.03), (variant,float((got-expected).abs().max()))


@pytest.mark.parametrize("variant,bits", WIDE_BITS)
def test_wide_subnormal_sources_preserve_nonzero_corrected_scale_and_finite_reconstruction(variant, bits):
    tiny = torch.tensor(1e-40).to(torch.bfloat16)
    source = torch.full((3,256), tiny.item(),dtype=torch.bfloat16)
    source[1,1::2] = 0
    source[1,128:] = 0
    source[2,1::2] = -tiny
    packed,metadata = encode(source.cuda(),bits,variant)
    norm = metadata.cpu().double()
    for row,payload,scales in zip(source.double().tolist(),packed.cpu().tolist(),norm.tolist()):
        indices = ref.unpack_indices_ref(bytes(payload),bits)
        for group in range(2):
            raw = math.hypot(*row[group*128:(group+1)*128])/math.sqrt(128)
            factor = (math.sqrt(128/math.fsum(ref.centroids(bits)[index]**2 for index in
                                            indices[group*128:(group+1)*128]))
                      if ref.norm_corrected(variant) else 1.)
            expected = raw*factor
            assert (scales[group] > 0) if raw else (scales[group] == 0)
            # These sources normalize to0/±1, with exact sum and power-of-two
            # mean. Original sqrt/multiply error is raw*u + half-subnormalULP;
            # correction contributes gamma127 energy plus four RN operations.
            u,q = 2**-24,2**-150
            gamma = 127*u/(1-127*u)
            correction_error = gamma+4*u if ref.norm_corrected(variant) else 0.
            bound = (raw*u+q)*factor*(1+correction_error)+expected*correction_error+q
            assert abs(scales[group]-expected) <= bound*(1+64*2**-53)
    decoded = decode(packed,metadata,bits,256).cpu()
    independent = torch.tensor([ref.dequantize_oracle(bytes(code),scale,bits,variant)
                                for code,scale in zip(packed.cpu().tolist(),norm.tolist())],dtype=torch.float32).to(torch.bfloat16)
    assert torch.equal(decoded,independent) and torch.isfinite(decoded).all()
    assert (decoded[0] != 0).any()


@pytest.mark.parametrize("variant,bits", WIDE_BITS)
def test_wide_nondefault_stream_and_graph_replay_keep_the_format_contract(variant, bits):
    # Uniform replay sources have exact dyadic normalized components, avoiding
    # midpoint-cancellation ambiguity while checking graph address/value reuse.
    test_codec_graph_replays_changed_inputs_without_hidden_norm_or_pointer_state(variant,bits)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        source = (torch.arange(3*256,device="cuda",dtype=torch.float32)/51).reshape(3,256).to(torch.bfloat16)
        packed,metadata = encode(source,bits,variant)
        restored = decode(packed,metadata,bits,256)
        event = torch.cuda.Event()
        event.record()
    torch.cuda.current_stream().wait_event(event)
    for row,payload,norm in zip(source.cpu().float().tolist(),packed.cpu().tolist(),metadata.cpu().tolist()):
        lower,upper = ref.index_envelope_oracle(row,bits,variant,rms=norm)
        indices = ref.unpack_indices_ref(bytes(payload),bits)
        assert all(low <= code <= high and high-low <= 1 for low,code,high in zip(lower,indices,upper))
    assert torch.equal(restored,ref.dequant_ref(packed,metadata,bits,variant).to(torch.bfloat16))


@pytest.mark.parametrize("magnitude",[0.,10000.,1e30])
def test_six_zero_and_large_finite_sources_keep_original_fp32_rms(magnitude):
    test_zero_and_large_finite_bf16_sources_do_not_use_fp16_norms("signed-iso128",6,magnitude)


def test_six_all_midpoint_neighbors_and_plane_boundaries_decode_exactly():
    test_exact_midpoints_and_fp32_neighbors_use_lower_tie(6)
    codes=[(13*i+i//127)%64 for i in range(512)]
    payload=ref.pack_indices_ref(codes,6)
    packed=torch.tensor(list(payload),dtype=torch.uint8,device="cuda").reshape(2,192)
    metadata=torch.tensor([[0.,.25],[1.,10000.]],dtype=torch.float32,device="cuda")
    # Independent whole-plane integer interpretation preserves every high2 bit.
    decoded_codes=[]
    for start in range(0,len(payload),96):
        low=int.from_bytes(payload[start:start+64],"little")
        high=int.from_bytes(payload[start+64:start+96],"little")
        decoded_codes.extend(((low>>(4*i))&15)|(((high>>(2*i))&3)<<4) for i in range(128))
    assert decoded_codes==codes and set(codes)==set(range(64))
    scales=metadata.cpu().flatten().tolist()
    values=[ref.centroids(6)[code]*scales[i//128] for i,code in enumerate(decoded_codes)]
    expected=torch.tensor(values,dtype=torch.float32).reshape(2,256).to(torch.bfloat16)
    assert torch.equal(decode(packed,metadata,6,256).cpu(),expected)


@pytest.fixture(scope="module")
def six_scaling_oracle_module():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).parents[1] / "test_rotorquant_six_scaling.py"
    spec = importlib.util.spec_from_file_location("rotorquant_independent_six_scaling",path)
    module = importlib.util.module_from_spec(spec)
    # Its dense stage helpers belong to the CPU-wide test module.
    import sys
    original = sys.path.copy()
    try:
        sys.path.insert(0,str(path.parent))
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = original
    return module


@pytest.mark.parametrize("variant",NORM_SIX_VARIANTS)
@pytest.mark.parametrize("magnitude",[0.,10000.,float(2**30)])
def test_norm_six_zero_and_large_finite_bf16_metadata_uses_c6_energy(variant,magnitude,wide_oracle_module):
    source = torch.full((3,256),magnitude,dtype=torch.bfloat16)
    packed,metadata = encode(source.cuda(),6,variant)
    assert metadata.dtype == torch.float32 and torch.isfinite(metadata).all()
    for row,payload,norm in zip(source.float().tolist(),packed.cpu().tolist(),metadata.cpu().tolist()):
        indices = ref.unpack_indices_ref(bytes(payload),6)
        lower,upper = ref.index_envelope_oracle(row,6,variant,rms=norm)
        assert all(lo <= code <= hi and hi-lo <= 1 for lo,code,hi in zip(lower,indices,upper))
        for start,scale in zip((0,128),norm):
            bound = wide_oracle_module.scale_interval(row[start:start+128],indices[start:start+128],
                                                     ref.CENTROIDS_6,True,max_scale_factor=32)
            assert bound[0] <= scale <= bound[1]
        if magnitude == 0:
            assert norm == [0.,0.] and indices == [31]*256
    back = decode(packed,metadata,6,256)
    assert torch.isfinite(back).all()
    assert torch.equal(back,ref.dequant_ref(packed,metadata,6,variant).to(torch.bfloat16))
    if magnitude == 0:
        assert back.eq(0).all()


@pytest.mark.parametrize("kind",["outlier","alpha-below","alpha-above"])
def test_outlier_six_rn_alpha_only_changes_indices_and_correction_uses_raw_rms(kind,six_scaling_oracle_module,
                                                                           wide_oracle_module):
    base = torch.tensor(six_scaling_oracle_module.source_fixture(kind),dtype=torch.float32)
    source = torch.stack([base,base*8]).repeat(1,2).to(torch.bfloat16)
    encoded = {}
    for variant in NORM_SIX_VARIANTS:
        packed,metadata = encode(source.cuda(),6,variant)
        encoded[variant] = packed.cpu()
        for row,payload,norm in zip(source.float().tolist(),packed.cpu().tolist(),metadata.cpu().tolist()):
            actual = ref.unpack_indices_ref(bytes(payload),6)
            low,high = ref.index_envelope_oracle(row,6,variant,rms=norm)
            assert all(lo <= code <= hi and hi-lo <= 1 for lo,code,hi in zip(low,actual,high))
            for begin,scale in zip((0,128),norm):
                # This bound contains rawRMS*sqrt(128/energy), deliberately
                # without alpha. An accidental final alpha multiplier fails.
                bound = wide_oracle_module.scale_interval(row[begin:begin+128],actual[begin:begin+128],
                                                         ref.CENTROIDS_6,True,max_scale_factor=32)
                assert bound[0] <= scale <= bound[1]
        assert torch.equal(decode(packed,metadata,6,256),ref.dequant_ref(packed,metadata,6,variant).to(torch.bfloat16))
    if kind == "outlier":
        assert not torch.equal(encoded[NORM_SIX_VARIANTS[0]],encoded[NORM_SIX_VARIANTS[1]])


@pytest.mark.parametrize("bits",[7,8],ids=["rotorquant7","rotorquant8"])
@pytest.mark.parametrize("magnitude",[0.,10000.,float(2**30)])
def test_high_precision_zero_and_large_finite_sources_keep_original_fp32_rms(bits,magnitude):
    test_zero_and_large_finite_bf16_sources_do_not_use_fp16_norms("signed-iso128",bits,magnitude)


@pytest.mark.parametrize("bits",[7,8],ids=["rotorquant7","rotorquant8"])
def test_high_precision_all_midpoints_and_all_plane_codes_decode_exactly(bits):
    test_exact_midpoints_and_fp32_neighbors_use_lower_tie(bits)
    codes = [(13*i+i//127)%(1 << bits) for i in range(512)]
    payload = ref.pack_indices_ref(codes,bits)
    if bits == 7:
        independent = []
        for start in range(0,len(payload),112):
            low = int.from_bytes(payload[start:start+64],"little")
            mid = int.from_bytes(payload[start+64:start+96],"little")
            high = int.from_bytes(payload[start+96:start+112],"little")
            independent.extend(((low>>(4*i))&15)|(((mid>>(2*i))&3)<<4)|(((high>>i)&1)<<6) for i in range(128))
    else:
        independent = list(payload)
    assert independent == codes and set(codes) == set(range(1 << bits))
    packed = torch.tensor(list(payload),dtype=torch.uint8,device="cuda").reshape(2,256*bits//8)
    metadata = torch.tensor([[0.,.25],[1.,10000.]],dtype=torch.float32,device="cuda")
    norms = metadata.cpu().flatten().tolist()
    original = [ref.centroids(bits)[code]*norms[i//128] for i,code in enumerate(independent)]
    expected = torch.tensor(original,dtype=torch.float32).reshape(2,256).to(torch.bfloat16)
    assert torch.equal(decode(packed,metadata,bits,256).cpu(),expected)

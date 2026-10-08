"""Exhaustive independent KV-pair CUDA gates; root controls execution.

Numeric assertions qualify implemented stored-basis arithmetic.
Trained quality, serving performance and resource codegen are separate gates.
Every descriptor is retained. No PPL cutoff and no fitted numeric tolerances.
"""
from __future__ import annotations

import itertools
import math
import os

import pytest
import torch
import triton
import triton.language as tl

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from tensorfold.families.qwen4_exp.kv_formats import FORMATS, KVPairFormat, get_pair  # noqa: E402
from tensorfold.families.qwen4_exp.cuda import attention, glue, kv_pair, rotorquant_ref as ref  # noqa: E402
from tensorfold.families.qwen4_exp.cuda.kvcache import KVCache  # noqa: E402
from tests.kv_pair_inverse_oracle import inverse_rounding_envelope, native_inverse_rounding_envelope  # noqa: E402
from tests.kv_pair_native_legacy_oracle import _canonical_legacy_chunk  # noqa: E402
from tests.kv_pair_failure_evidence import save_failure_snapshot  # noqa: E402

PAIRS = tuple(KVPairFormat(k, v) for k, v in itertools.product(FORMATS, repeat=2))
H, HK, D = 24, 2, 256


def pair_id(pair):
    return f"K-{pair.key_dtype}__V-{pair.value_dtype}"


def traits(pair):
    return {"k_bits": 0 if pair.key.bits == 16 else pair.key.bits, "k_codec": pair.key.codec,
            "v_bits": 0 if pair.value.bits == 16 else pair.value.bits, "v_codec": pair.value.codec}


def dump_failure(name, pair, tensors, error):
    """Keep exact operands before fail; the owned root runner supplies this path."""
    directory = os.environ.get("TENSORFOLD_KV_PAIR_FAILURE_DIR")
    if directory:
        try:
            save_failure_snapshot(
                directory, name, pair.identity, pair_id(pair), os.environ["PYTEST_CURRENT_TEST"], tensors,
                torch.save, torch.is_tensor,
                lambda tensor: tensor.detach().to(device="cpu", copy=True, memory_format=torch.contiguous_format),
            )
        except BaseException as secondary:
            error.add_note(f"failure evidence could not be saved: {secondary!r}")
    raise error


def exact(name, pair, actual, expected, **evidence):
    integer = torch.int16 if actual.dtype == torch.bfloat16 else torch.int32
    if not torch.equal(actual.view(integer), expected.view(integer)):
        dump_failure(name, pair, {"actual": actual, "expected": expected, **evidence},
                     AssertionError(f"{name} bits differ for {pair_id(pair)}"))


def h32_independent(x):
    """Five explicit rounded binary32 butterfly stages, then RN32 scale."""
    shape = x.shape
    x = x.float().reshape(-1, 32)
    for stride in (1, 2, 4, 8, 16):
        split = x.reshape(-1, 32 // (2 * stride), 2, stride)
        a, b = split[:, :, 0], split[:, :, 1]
        x = torch.stack((a + b, a - b), 2).reshape(-1, 32)
    return (x * torch.tensor(1 / math.sqrt(32), dtype=torch.float32, device=x.device)).reshape(shape)


def decoded_operand(format, code, scale):
    """Independent byte decoding with explicit FP32 multiply then one BF16 round.

    Rotor unpack uses the independent byte-loop oracle, not a kernel helper.
    Native INT8 signed bytes and low-even INT4 nibbles are decoded here directly.
    """
    if format.bits == 16:
        return code.clone()
    shape = (*code.shape[:-1], code.shape[-1] * 8 // format.bits)
    if format.codec:
        flat = code.cpu().reshape(-1, code.shape[-1]).tolist()
        indices = [ref.unpack_indices_ref(bytes(row), format.bits) for row in flat]
        index = torch.tensor(indices, dtype=torch.int64, device=code.device).reshape(shape)
        book = torch.tensor(ref.centroids(format.bits), dtype=torch.float32, device=code.device)
        out = book[index].reshape(*scale.shape, 128) * scale[..., None]
    else:
        if format.bits == 8:
            coordinate = code.float() + 0.5
            multiplier = 1 / 128
        else:
            byte = code.to(torch.int32)
            coordinate = torch.stack(((byte & 15).float(), ((byte >> 4) & 15).float()), -1).reshape(shape) - 7.5
            multiplier = 1 / 8
        out = coordinate.reshape(*scale.shape, 32) * scale.float()[..., None]
        out = out * multiplier
    return out.reshape(shape).bfloat16()


@triton.jit
def _capture_operand(DATA, SCALE, OUT, N: tl.constexpr, HK_: tl.constexpr, D_: tl.constexpr,
                     BITS: tl.constexpr, CODEC: tl.constexpr):
    row = tl.program_id(0) * 64 + tl.arange(0, 64)
    head = tl.program_id(1)
    operand = kv_pair.load_side(DATA, SCALE, row, row < N, head, HK_, D_, 64, BITS, CODEC)
    address = (row[:, None] * HK_ + head) * D_ + tl.arange(0, D_)[None, :]
    tl.store(OUT + address, operand, mask=(row < N)[:, None])


@triton.jit
def _canonical_chunk(Q, K, V, POS, PO, PM, PL, IDS, NK, SPARSE,
                     H_: tl.constexpr, HK_: tl.constexpr, D_: tl.constexpr, G: tl.constexpr,
                     CH: tl.constexpr, NCH: tl.constexpr, IDW: tl.constexpr,
                     SCALE: tl.constexpr, QSA: tl.constexpr):
    """Independent canonical64 arithmetic; no production tile/chunk/decode reuse."""
    row, head, chunk = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    n = tl.load(POS) + row + 1
    sparse = False
    if QSA:
        sparse = tl.load(SPARSE + row) != 0
        n = tl.where(sparse, tl.load(NK + row), n)
    start = chunk * CH
    if start < n:
        g = tl.arange(0, 16)
        d = tl.arange(0, D_)
        q = tl.load(Q + (row * H_ + head * G + g[:, None]) * D_ + d[None, :],
                    mask=g[:, None] < G, other=0.)
        maximum = tl.full((16,), float("-inf"), tl.float32)
        denominator = tl.zeros((16,), tl.float32)
        numerator = tl.zeros((16, D_), tl.float32)
        for tile in range(0, tl.cdiv(tl.minimum(n - start, CH), 64)):
            position = start + tile * 64 + tl.arange(0, 64)
            valid = position < n
            if QSA:
                if sparse:
                    position = tl.load(IDS + row * IDW + position, mask=valid, other=0)
            address = (position[:, None].to(tl.int64) * HK_ + head) * D_ + d[None, :]
            k = tl.load(K + address, mask=valid[:, None], other=0.)
            v = tl.load(V + address, mask=valid[:, None], other=0.)
            score = tl.dot(q, tl.trans(k)).to(tl.float32) * SCALE
            score = tl.where(valid[None, :], score, float("-inf"))
            tile_max = tl.max(score, 1)
            active = tile_max != float("-inf")
            next_max = tl.where(active, tl.maximum(maximum, tile_max), maximum)
            alpha = tl.where(active, tl.where(maximum == float("-inf"), 0., tl.exp(maximum - next_max)), 1.)
            probability = tl.where(valid[None, :] & active[:, None], tl.exp(score - next_max[:, None]), 0.)
            numerator = numerator * alpha[:, None] + tl.dot(probability.to(tl.bfloat16), v)
            denominator = denominator * alpha + tl.sum(probability, 1)
            maximum = next_max
        address = (row * NCH + chunk) * H_ + head * G + g
        tl.store(PO + address[:, None] * D_ + d[None, :], numerator, mask=g[:, None] < G)
        tl.store(PM + address, maximum, mask=g < G)
        tl.store(PL + address, denominator, mask=g < G)


@triton.jit
def _canonical_merge(PO, PM, PL, POS, RAW, NK, SPARSE, H_: tl.constexpr, HK_: tl.constexpr,
                     D_: tl.constexpr, G: tl.constexpr, CH: tl.constexpr, NCH: tl.constexpr,
                     QSA: tl.constexpr):
    row, head = tl.program_id(0), tl.program_id(1)
    n = tl.load(POS) + row + 1
    if QSA:
        n = tl.where(tl.load(SPARSE + row) != 0, tl.load(NK + row), n)
    g, d = tl.arange(0, 16), tl.arange(0, D_)
    maximum = tl.full((16,), float("-inf"), tl.float32)
    denominator = tl.zeros((16,), tl.float32)
    numerator = tl.zeros((16, D_), tl.float32)
    for chunk in range(0, tl.cdiv(n, CH)):
        address = (row * NCH + chunk) * H_ + head * G + g
        m = tl.load(PM + address, mask=g < G, other=float("-inf"))
        weight = tl.load(PL + address, mask=g < G, other=0.)
        o = tl.load(PO + address[:, None] * D_ + d[None, :], mask=g[:, None] < G, other=0.)
        active = weight > 0.
        next_max = tl.where(active, tl.maximum(maximum, m), maximum)
        a = tl.where(active, tl.where(maximum == float("-inf"), 0., tl.exp(maximum - next_max)), 1.)
        b = tl.where(active, tl.exp(m - next_max), 0.)
        numerator = numerator * a[:, None] + o * b[:, None]
        denominator = denominator * a + weight * b
        maximum = next_max
    tl.store(RAW + (row * H_ + head * G + g[:, None]) * D_ + d[None, :],
             numerator / denominator[:, None], mask=g[:, None] < G)


def capture(format, code, scale):
    n, hk = code.shape[:2]
    out = torch.empty((n, hk, D), dtype=torch.bfloat16, device="cuda")
    _capture_operand[(triton.cdiv(n, 64), hk)](code, scale, out, n, hk, D,
        0 if format.bits == 16 else format.bits, format.codec, num_warps=4)
    expected = decoded_operand(format, code, scale)
    assert torch.equal(out.view(torch.int16), expected.view(torch.int16))
    return out


def make_projection(rows, seed=618):
    pw = H * 2 * D + HK * 2 * D + 2 * 128
    generator = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randn((rows, pw), device="cuda", generator=generator).bfloat16()


def prep(pair, projection, *, capacity=None, position=0, delta=0):
    rows = projection.shape[0]
    capacity = rows + position if capacity is None else capacity
    cache = KVCache(capacity, HK, D, "cuda", pair=pair)
    q = torch.empty((rows, H, D), device="cuda", dtype=torch.bfloat16)
    iq = torch.empty((rows, 1, 128), device="cuda", dtype=torch.bfloat16)
    index = torch.zeros((capacity, 128), device="cuda", dtype=torch.bfloat16)
    pos = torch.tensor([position], device="cuda", dtype=torch.int32)
    rotary = torch.tensor([delta], device="cuda", dtype=torch.int32)
    inv = (1e7 ** (-torch.arange(32, device="cuda", dtype=torch.float64) / 32)).float()
    scale, index_scale = torch.ones(D, device="cuda"), torch.ones(128, device="cuda")
    status = None if pair.symmetric and not pair.key.codec else torch.tensor([0, -2], device="cuda", dtype=torch.int32)
    glue.attn_prep(projection, pos, scale, scale, index_scale, inv, q, cache.k, cache.v, iq, index,
                   1e-6, q_heads=H, kv_heads=HK, head_dim=D, index_heads=1, index_dim=128,
                   ks=cache.ks, vs=cache.vs, delta=rotary, status=status, **traits(pair))
    return q, iq, index, cache, status


@pytest.mark.parametrize("pair", PAIRS, ids=pair_id)
def test_pair_prep_value_choice_does_not_change_query_or_index(pair):
    projection = make_projection(3)
    q, iq, index, candidate, status = prep(pair, projection, delta=524284)
    control_pair = KVPairFormat(pair.key, pair.key)
    q0, iq0, index0, control, status0 = prep(control_pair, projection, delta=524284)
    exact("query-key-only", pair, q, q0)
    exact("index-query", pair, iq, iq0)
    exact("raw-index", pair, index, index0)
    assert torch.equal(candidate.k.view(torch.uint8), control.k.view(torch.uint8))
    assert torch.equal(candidate.ks.view(torch.uint8), control.ks.view(torch.uint8))
    for guard in (status, status0):
        if guard is not None:
            assert guard.cpu().tolist() == [0, -2]


@pytest.mark.parametrize("pair", PAIRS, ids=pair_id)
@pytest.mark.parametrize("length,rows,sparse", [(17, 1, False), (65, 1, False), (513, 1, False),
                                               (513, 17, False), (513, 17, True),
                                               (1536, 256, False), (1536, 256, True)])
def test_all_pairs_native64_actual_operand_partials(pair, length, rows, sparse):
    capacity = max(length + 4, 640 if sparse else length)
    projected = make_projection(capacity)
    q, _, _, cache, status = prep(pair, projected, capacity=capacity)
    if status is not None:
        assert status.cpu().tolist() == [0, -2]
    key = capture(pair.key, cache.k, cache.ks)
    value = capture(pair.value, cache.v, cache.vs)
    query = q[length - rows:length].contiguous()
    pos = torch.tensor([length - rows], device="cuda", dtype=torch.int32)
    actual = attention.AttnScratch(rows, H, D, capacity, "cuda", budget=128 if sparse else 2048)
    oracle = attention.AttnScratch(rows, H, D, capacity, "cuda", budget=128 if sparse else 2048)
    if sparse:
        for row in range(rows):
            n = length - rows + row + 1
            ids = torch.arange(0, n, 5, device="cuda", dtype=torch.int32)[:125]
            for target in (actual, oracle):
                target.ids[row, :ids.numel()] = ids
                target.nk[row] = ids.numel()
                target.sparse[row] = 1
    got = attention.attention(query, cache.k, cache.v, pos, actual, rows, .0625,
                              ks=cache.ks, vs=cache.vs, **traits(pair))
    legacy_native = pair.symmetric and pair.key.codec == 0 and pair.key.bits in (4, 8)
    if legacy_native:
        # This oracle qualifies finite attention operands. The separate layout
        # bit test covers every encoding without doing floating arithmetic.
        assert torch.isfinite(query).all() and torch.isfinite(key).all() and torch.isfinite(value).all()
        _canonical_legacy_chunk[(rows, HK, oracle.nch)](query, key, value, pos, oracle.po, oracle.pm, oracle.pl,
            oracle.ids, oracle.nk, oracle.sparse, H, HK, D, H // HK, attention.CHUNK, oracle.nch,
            oracle.idw, .0625, oracle.qsa, num_warps=4, num_stages=1, enable_fp_fusion=False)
    else:
        _canonical_chunk[(rows, HK, oracle.nch)](query, key, value, pos, oracle.po, oracle.pm, oracle.pl,
            oracle.ids, oracle.nk, oracle.sparse, H, HK, D, H // HK, attention.CHUNK, oracle.nch,
            oracle.idw, .0625, oracle.qsa, num_warps=4, num_stages=1)
    for field in ("po", "pm", "pl"):
        exact("chunk-" + field, pair, getattr(actual, field), getattr(oracle, field), query=query, key=key, value=value)
    raw = torch.empty_like(got, dtype=torch.float32)
    _canonical_merge[(rows, HK)](oracle.po, oracle.pm, oracle.pl, pos, raw, oracle.nk, oracle.sparse,
        H, HK, D, H // HK, attention.CHUNK, oracle.nch, oracle.qsa, num_warps=4)
    if pair.value.bits == 16:
        exact("merge-raw", pair, got, raw.bfloat16(), query=query, key=key, value=value)
    else:
        # Exact partials above isolate attention. The inverse has a separate
        # analytically bounded staged FP32 contract, not exact Torch equality.
        assert torch.isfinite(got).all()
        assert_bf16_inverse_envelope(pair, raw, got)




def assert_bf16_inverse_envelope(pair, raw, actual):
    format = pair.value
    if format.codec:
        before, bound = [], []
        for row in raw.cpu().reshape(-1, D).tolist():
            center, radius = inverse_rounding_envelope(row, format.rotor.variant)
            before.append(center)
            bound.append(radius)
        center = torch.tensor(before, dtype=torch.float64).reshape(raw.shape)
        error = torch.tensor(bound, dtype=torch.float64).reshape(raw.shape)
    else:
        before, bound = [], []
        for row in raw.cpu().reshape(-1, D).tolist():
            center, radius = native_inverse_rounding_envelope(row)
            before.append(center)
            bound.append(radius)
        center = torch.tensor(before, dtype=torch.float64).reshape(raw.shape)
        error = torch.tensor(bound, dtype=torch.float64).reshape(raw.shape)
    negative = torch.full_like(center, -math.inf)
    positive = torch.full_like(center, math.inf)
    # Outward binary64 endpoints include the bound evaluation's final RN;
    # final BF16 conversion is explicit and monotone.
    lower = torch.nextafter(center - error, negative).bfloat16().double()
    upper = torch.nextafter(center + error, positive).bfloat16().double()
    observed = actual.cpu().double()
    if not bool(((observed >= lower) & (observed <= upper)).all()):
        dump_failure("merge-value-only-envelope", pair, {"raw": raw, "actual": actual,
                                                       "center": center, "radius": error},
                     AssertionError("value inverse violates its derived FP32/BF16 envelope"))






@triton.jit
def _guard_and_store(X, DATA, SCALE, STATUS, D_: tl.constexpr, BITS: tl.constexpr,
                     CODEC: tl.constexpr, ERROR: tl.constexpr):
    x = tl.load(X + tl.arange(0, D_)).to(tl.float32)
    x = kv_pair.source_guard(x, STATUS, ERROR, 42)
    kv_pair.store_side(x, DATA, SCALE, 0, STATUS, D_, BITS, CODEC, True, 42)


@pytest.mark.parametrize("format", FORMATS, ids=lambda f: f.name)
@pytest.mark.parametrize("side,error", [("K", 2), ("V", 4)])
@pytest.mark.parametrize("failure", ["nan", "inf", "over-source-bound"])
def test_every_side_source_guard_has_actual_failed_status(format, side, error, failure):
    pair = KVPairFormat(format, format)
    cache = KVCache(1, 1, D, "cuda", pair=pair)
    x = torch.ones(D, device="cuda", dtype=torch.bfloat16)
    x[9] = {"nan": float("nan"), "inf": float("inf"), "over-source-bound": float(2**31)}[failure]
    status = torch.tensor([0, -2], device="cuda", dtype=torch.int32)
    code, scale = (cache.k, cache.ks) if side == "K" else (cache.v, cache.vs)
    _guard_and_store[(1,)](x, code, scale, status, D, 0 if format.bits == 16 else format.bits,
                          format.codec, error, num_warps=2)
    assert status.cpu().tolist() == [error, 42]
    assert torch.isfinite(scale).all()


@pytest.mark.parametrize("format", [f for f in FORMATS if not f.codec and f.bits != 16], ids=lambda f: f.name)
def test_mixed_native_scale_fp16_overflow_is_contained_and_underflow_is_legacy(format):
    pair = KVPairFormat(format, get_pair("bf16").key)
    cache = KVCache(1, 1, D, "cuda", pair=pair)
    status = torch.tensor([0, -2], device="cuda", dtype=torch.int32)
    x = torch.full((D,), 65504., device="cuda", dtype=torch.bfloat16)
    _guard_and_store[(1,)](x, cache.k, cache.ks, status, D, format.bits, 0, 2, num_warps=2)
    assert status.cpu().tolist() == [8, 42]
    assert torch.isfinite(cache.ks).all() and torch.all(cache.ks == 0)
    status.copy_(torch.tensor([0, -2], device="cuda", dtype=torch.int32))
    x.fill_(torch.finfo(torch.bfloat16).tiny)
    _guard_and_store[(1,)](x, cache.k, cache.ks, status, D, format.bits, 0, 2, num_warps=2)
    assert status.cpu().tolist() == [0, -2]
    assert torch.all(cache.ks == 0)


@pytest.mark.parametrize("dtype", ["bf16", "int8", "int4"])
def test_symmetric_native_explicit_traits_keep_original_dispatch_and_all_bits(dtype, monkeypatch):
    pair = get_pair(dtype)
    projection = make_projection(65)
    q, _, _, cache, _ = prep(pair, projection)
    pos = torch.tensor([64], device="cuda", dtype=torch.int32)
    old, explicit = [attention.AttnScratch(1, H, D, 65, "cuda") for _ in range(2)]
    recorded = []
    original_chunks = attention._chunks
    class Recorder:
        def __getitem__(self, grid):
            launch = original_chunks[grid]
            def wrapped(*args, **kwargs):
                recorded.append(dict(kwargs))
                return launch(*args, **kwargs)
            return wrapped
    monkeypatch.setattr(attention, "_chunks", Recorder())
    original = attention.attention(q[-1:].contiguous(), cache.k, cache.v, pos, old, 1, .0625,
        ks=cache.ks, vs=cache.vs, bits=0 if pair.key.bits == 16 else pair.key.bits)
    candidate = attention.attention(q[-1:].contiguous(), cache.k, cache.v, pos, explicit, 1, .0625,
        ks=cache.ks, vs=cache.vs, **traits(pair))
    assert len(recorded) == 2
    assert all(not any(k in args for k in ("K_BITS", "K_CODEC", "V_BITS", "V_CODEC")) for args in recorded)
    for field in ("po", "pm", "pl"):
        exact("native-legacy-" + field, pair, getattr(explicit, field), getattr(old, field))
    exact("native-legacy-output", pair, candidate, original)


@pytest.fixture(scope="module")
def pair_model():
    from test_flashnext_forward import _model
    return _model(seed=71)


@pytest.mark.parametrize("pair", PAIRS, ids=pair_id)
def test_each_pair_prefix_clone_growth_and_continuation_bits(pair, pair_model):
    from tensorfold.families.qwen4_exp.cuda.decode import Engine, prefill
    from tensorfold.families.qwen4_exp.cuda.forward import forward, commit
    w = pair_model
    options = dict(capacity=128, max_rows=4, prefill_rows=16, kv_pair=pair)
    source, target = Engine(w, **options), Engine(w, **options)
    prompt = [5, 17, 99, 250, 7, 64, 300, 11]
    prefill(source, prompt, None)
    target.st.copy_prefix(source.st, source.st.pos, source.st.mtp_len)
    target.st.restore(source.st.snapshot())
    clone = Engine(w, **options)
    clone.st = source.st.clone()
    source.st.resize(32)
    before = source.st.cache_bytes()
    added = source.st.ensure(40, step=64)
    assert source.st.capacity == 64
    assert source.st.cache_bytes() - before == added and added > 0
    for engine in (target, clone):
        assert engine.st.kv_identity == source.st.kv_identity
        for original, copied in zip([*source.st.kc, source.st.mtp_kc], [*engine.st.kc, engine.st.mtp_kc], strict=True):
            assert original.pair == copied.pair == pair
            keep = source.st.mtp_len if original is source.st.mtp_kc else source.st.pos
            for a, b in zip((original.k, original.v, original.ks, original.vs),
                            (copied.k, copied.v, copied.ks, copied.vs), strict=True):
                n = 1 if a.shape == (1,) else keep
                assert torch.equal(a[:n].view(torch.uint8), b[:n].view(torch.uint8))
    for token in (17, 400, 7):
        outputs = [forward(w, engine.st, engine.buf, [token]).clone() for engine in (source, target, clone)]
        exact("prefix-continuation", pair, outputs[1], outputs[0])
        exact("clone-continuation", pair, outputs[2], outputs[0])
        for engine in (source, target, clone):
            commit(w, engine.st, engine.buf, 1, 1)


@pytest.mark.parametrize("pair", PAIRS, ids=pair_id)
def test_each_pair_graph_mtp_and_nondefault_stream_same_pair_outputs(pair, pair_model):
    from tensorfold.engine.exact_sampling import Sampling
    from tensorfold.families.qwen4_exp.cuda.decode import Engine, mtp_decode, prefill, serial_decode
    w = pair_model
    options = dict(capacity=128, max_rows=4, prefill_rows=16, kv_pair=pair)
    sampling = Sampling(seed=592, top_k=20, top_p=.95)
    prompt = [5, 17, 99, 7, 64, 300, 11]
    serial = Engine(w, **options)
    first = prefill(serial, prompt, sampling)
    expected = serial_decode(serial, first, 8, sampling).tokens
    graph = Engine(w, graphs=True, **options)
    first = prefill(graph, prompt, sampling)
    assert serial_decode(graph, first, 8, sampling).tokens == expected
    for engine in (serial, graph):
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            first = prefill(engine, prompt, sampling)
            got = mtp_decode(engine, first, 8, sampling, depth=3, confidence=0.)
        torch.cuda.current_stream().wait_stream(stream)
        assert got.tokens == expected
        engine.st.kv_check()


@pytest.mark.parametrize("pair", PAIRS, ids=pair_id)
def test_each_pair_yarn_vision_sparse_multi_uses_same_pair_solo_contract(pair):
    from dataclasses import replace
    from types import SimpleNamespace
    from test_flashnext_forward import _model
    from test_flashnext_vision import _image
    from test_flashnext_yarn import _policy
    from tensorfold.cuda.streams import Stream
    from tensorfold.engine.exact_sampling import Sampling
    from tensorfold.families.qwen4_exp.cuda.decode import Engine, mtp_decode, prefill, serial_decode
    from tensorfold.families.qwen4_exp.cuda.multi import MultiDecoder
    w = _model(seed=72)
    policy = _policy()
    w.cfg = replace(w.cfg, rope=policy, index_budget=8)
    w.inv_freq = policy.inverse_frequencies(torch).cuda()
    prompt = [5, 17, 17, 17, 17] + list(range(20, 56))
    visual = _image(w, prompt, 41)
    sampling = Sampling(seed=59, top_k=20, top_p=.95)
    expected = []
    for image in (None, visual):
        engine = Engine(w, capacity=128, max_rows=4, prefill_rows=16, kv_pair=pair)
        first = prefill(engine, prompt, sampling, vision=image)
        expected.append(serial_decode(engine, first, 8, sampling).tokens)
        first = prefill(engine, prompt, sampling, vision=image)
        assert mtp_decode(engine, first, 8, sampling, depth=3, confidence=0.).tokens == expected[-1]
    decoder = MultiDecoder(w, slots=4, capacity=128, depth=3, confidence=0., kv_pair=pair,
        vision=SimpleNamespace(encode=lambda prepared, ids: prepared), prefill_rows=16)
    streams = [Stream(prompt, 8, sampling, stop_eos=False, vision=image) for image in (None, visual, None, visual)]
    for stream in streams:
        decoder.admit(stream)
    for _ in range(64):
        if not decoder.live():
            break
        decoder.finish(decoder.round())
        assert all(stream.error is None for stream in streams)
    assert not decoder.live()
    assert [stream.out for stream in streams] == expected * 2


@pytest.mark.parametrize("pair", PAIRS, ids=pair_id)
def test_every_pair_expert_cache_composition_is_exact_same_pair(pair, pair_model):
    from test_flashnext_ram_experts import _offload
    from tensorfold.engine.exact_sampling import Sampling
    from tensorfold.families.qwen4_exp.cuda.decode import Engine, mtp_decode, prefill, serial_decode
    w = pair_model
    options = dict(capacity=128, max_rows=4, prefill_rows=8, kv_pair=pair)
    sampling = Sampling(seed=17, top_k=20, top_p=.95)
    prompt = [5, 17, 99, 7, 64, 300, 11]
    resident = Engine(w, **options)
    first = prefill(resident, prompt, sampling)
    expected = serial_decode(resident, first, 8, sampling).tokens
    offloaded, cache = _offload(w, slots=2)
    try:
        cached = Engine(offloaded, graphs=True, **options)
        assert cached.graphs is None
        assert prefill(cached, prompt, sampling) == first
        assert serial_decode(cached, first, 8, sampling).tokens == expected
        assert prefill(cached, prompt, sampling) == first
        assert mtp_decode(cached, first, 8, sampling, depth=2, confidence=0.).tokens == expected
    finally:
        cache.close()


@pytest.mark.parametrize("format", [f for f in FORMATS if f.codec], ids=lambda f: f.name)
def test_rotor_forbidden_fp32_erasure_poison_is_not_native_underflow(format):
    cache = KVCache(1, 1, D, "cuda", format.name)
    status = torch.tensor([0, -2], device="cuda", dtype=torch.int32)
    source = torch.zeros(D, device="cuda", dtype=torch.float32)
    source[0] = 2**-149
    _guard_and_store[(1,)](source, cache.k, cache.ks, status, D, format.bits, format.codec, 2, num_warps=2)
    assert status.cpu().tolist() == [8, 42]
    assert torch.isfinite(cache.ks).all()


def native_encoder_independent(source, bits):
    rotated = h32_independent(source.cpu().reshape(-1, 32))
    maximum = rotated.abs().amax(-1) + torch.tensor(1e-10, dtype=torch.float32)
    unit = rotated * (torch.ones_like(maximum) / maximum)[:, None]
    count = 128 if bits == 8 else 8
    codes = (torch.floor(unit * count) + count).clamp(0, 2 * count - 1).to(torch.int32)
    if bits == 8:
        payload = (codes - 128).to(torch.int8)
    else:
        payload = (codes[:, 0::2] | (codes[:, 1::2] << 4)).to(torch.uint8)
    data = payload.reshape(*source.shape[:-1], source.shape[-1] * bits // 8)
    scale = maximum.half().reshape(*source.shape[:-1], source.shape[-1] // 32)
    return data, scale


@pytest.mark.parametrize("format", FORMATS, ids=lambda f: f.name)
def test_each_side_encoder_against_native_post_rope_source_and_independent_bounds(format):
    projection = make_projection(3, seed=775)
    native = get_pair("bf16")
    q0, iq0, index0, baseline, _ = prep(native, projection, delta=524284)
    pair = KVPairFormat(format, format)
    q, iq, index, cache, status = prep(pair, projection, delta=524284)
    if status is not None:
        assert status.cpu().tolist() == [0, -2]
    exact("encoder-index-query", pair, iq, iq0)
    exact("encoder-index-raw", pair, index, index0)
    if format.bits == 16:
        expected_query = q0
    elif format.codec:
        expected_query = ref.rotate_ref(q0.float(), format.rotor.variant).bfloat16()
    else:
        expected_query = h32_independent(q0.float()).bfloat16()
    exact("encoder-key-query-transform", pair, q, expected_query)
    for label, source, data, metadata in (("K", baseline.k, cache.k, cache.ks),
                                         ("V", baseline.v, cache.v, cache.vs)):
        if format.bits == 16:
            exact("encoder-bf16-" + label, pair, data, source)
            continue
        if not format.codec:
            expected, scales = native_encoder_independent(source, format.bits)
            assert torch.equal(data.cpu().view(torch.uint8), expected.view(torch.uint8))
            assert torch.equal(metadata.cpu().view(torch.uint8), scales.view(torch.uint8))
            continue
        source_rows = source.cpu().float().reshape(-1, D).tolist()
        payload_rows = data.cpu().reshape(-1, D * format.bits // 8).tolist()
        scale_rows = metadata.cpu().reshape(-1, D // 128).tolist()
        for raw, packed, scales in zip(source_rows, payload_rows, scale_rows, strict=True):
            codes = ref.unpack_indices_ref(bytes(packed), format.bits)
            # Byte-loop reconstruction guards every split-plane and nibble;
            # no production pack/unpack helper participates in this oracle.
            assert ref.pack_indices_ref(codes, format.bits) == bytes(packed)
            low, high = ref.index_envelope_oracle(raw, format.bits, format.rotor.variant, rms=scales)
            if not all(a <= code <= b for a, code, b in zip(low, codes, high, strict=True)):
                dump_failure("encoder-index-envelope-" + label, pair,
                    {"source": source, "payload": data, "scale": metadata},
                    AssertionError("centroid indices violate independent FP32 normalization/rotation bound"))
            _, _, ideal_scale, scale_radius = ref.rounding_envelope_oracle(raw, format.rotor.variant,
                                                                           rms=scales, bits=format.bits)
            # The reference validates actual metadata against the derived
            # norm interval as part of its conditional index proof. Keep the
            # explicit per-group comparison as independent evidence too.
            assert all(abs(actual - center) <= radius for actual, center, radius in
                       zip(scales, ideal_scale, scale_radius, strict=True))


@pytest.mark.parametrize("pair", [pair for pair in PAIRS if not pair.symmetric or pair.key.codec], ids=pair_id)
@pytest.mark.parametrize("failure,bit", [("Q", 1), ("K", 2), ("V", 4), ("index-query", 16), ("index-key", 16)])
def test_guarded_pair_prep_reports_each_actual_source_and_keeps_failed_frame_finite(pair, failure, bit):
    projection = make_projection(3, seed=884)
    offsets = {"Q": 9, "K": H * 2 * D + 9, "V": H * 2 * D + HK * D + 9,
               "index-query": H * 2 * D + HK * 2 * D + 9,
               "index-key": H * 2 * D + HK * 2 * D + 128 + 9}
    projection[1, offsets[failure]] = float("inf") if failure == "K" else float("nan")
    query, iq, index, cache, status = prep(pair, projection)
    assert status.cpu().tolist() == [bit, 0]
    assert all(torch.isfinite(tensor).all() for tensor in (query, iq, index, cache.ks, cache.vs))

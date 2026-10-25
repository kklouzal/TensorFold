"""The ordinary family adapter keeps the selected Q4 basis and cache ownership."""

from types import SimpleNamespace

import pytest
import torch

from tensorfold.cuda.sm70 import family_ops as ops
from tensorfold.families.qwen4_exp.cuda import rotorquant_ref as ref


def fixture(kb=4, vb=4, *, kf=None, vf=None, rows=3, live=2, capacity=9, position=2):
    torch.manual_seed(192)
    qh, kh, ih, dim, iw = 3, 1, 2, 128, 128
    projection = torch.randn(rows, 2 * qh * dim + 2 * kh * dim + (ih + 1) * iw).half() * .2

    def side(bits):
        payload = torch.full((capacity, kh, dim // 2 if bits == 4 else dim), 9,
                             dtype={0: torch.float16, 8: torch.int8, 4: torch.uint8}[bits])
        scale = None if not bits else torch.full(
            (capacity, kh, dim // (128 if bits == 4 else 32)), .25,
            dtype=torch.float32 if bits == 4 else torch.float16)
        return payload, scale

    k, ks = side(kb)
    v, vs = side(vb)
    return SimpleNamespace(p=projection, pos=torch.tensor([position], dtype=torch.int32),
                           live=torch.tensor([live], dtype=torch.int32), status=torch.tensor([0, -2], dtype=torch.int32),
                           qw=torch.ones(dim), kw=torch.ones(dim), iw=torch.ones(iw),
                           freq=torch.tensor([.01 * i for i in range(32)]),
                           q=torch.zeros(rows, qh, dim).half(), k=k, v=v, ks=ks, vs=vs,
                           iq=torch.zeros(rows, ih, iw).half(), ik=torch.full((capacity, iw), 9.).half(),
                           kb=kb, vb=vb, kf=4 if kb == 4 and kf is None else kf or 0,
                           vf=4 if vb == 4 and vf is None else vf or 0,
                           rows=rows, qh=qh, kh=kh, ih=ih, dim=dim, indexdim=iw)


def prepare(f):
    ops.attn_prep(f.p, f.pos, f.qw, f.kw, f.iw, f.freq, f.q, f.k, f.v, f.iq, f.ik, 1e-6,
                  q_heads=f.qh, kv_heads=f.kh, head_dim=f.dim, index_heads=f.ih,
                  index_dim=f.indexdim, ks=f.ks, vs=f.vs, k_bits=f.kb, k_codec=f.kf,
                  v_bits=f.vb, v_codec=f.vf, valid_rows=f.live, status=f.status, layer=5)


@pytest.mark.parametrize("kb,vb", [(4, 4), (4, 8), (8, 4), (4, 0), (0, 4)])
def test_cpu_family_preparation_uses_each_declared_basis_and_preserves_idle_rows(kb, vb):
    f = fixture(kb, vb)
    original = {name: getattr(f, name).clone() for name in ("k", "v", "ik")}
    prepare(f)
    positions = f.pos.long() + torch.arange(f.rows)
    query = f.p[:, :f.qh * 2 * f.dim].reshape(f.rows, f.qh, 2, f.dim)[:, :, 0]
    query = ops.rotary(ops.rms(query, f.qw, 1e-6), f.freq, positions)
    expected_q = ref.rotate_ref(query, "signed-iso128").half() if kb == 4 else (
        ops.h32(query).half() if kb else query)
    assert torch.equal(f.q[:2], expected_q[:2])
    value = f.p[:, 2 * f.qh * f.dim + f.kh * f.dim:
                2 * f.qh * f.dim + 2 * f.kh * f.dim].reshape(f.rows, f.kh, f.dim)
    if vb == 4:
        codes, scales = ref.quantize_ref(value[:2], 4, "signed-iso128")
        assert torch.equal(f.v[2:4], codes) and torch.equal(f.vs[2:4], scales)
    elif vb == 8:
        codes, scales = ops.quantize_int8(value[:2])
        assert torch.equal(f.v[2:4], codes) and torch.equal(f.vs[2:4], scales)
    else:
        assert torch.equal(f.v[2:4], value[:2])
    for name in ("k", "v", "ik"):
        actual = getattr(f, name)
        assert torch.equal(actual[:2], original[name][:2])
        assert torch.equal(actual[4:], original[name][4:])
    assert f.status.tolist() == [0, -2]


def test_dynamic_last_live_row_does_not_get_overwritten_by_out_of_bounds_padding():
    f = fixture(4, 0, capacity=4, position=3, live=1)
    prepare(f)
    start = 2 * f.qh * f.dim + f.kh * f.dim
    expected = f.p[0, start:start + f.kh * f.dim].reshape(f.kh, f.dim)
    assert torch.equal(f.v[3], expected)
    assert f.status.tolist() == [0, -2]
    assert not torch.equal(f.ik[3], torch.full_like(f.ik[3], 9))


def test_reference_rotor_encoding_preserves_zero_live_cache_and_reports_working_overflow():
    f = fixture(live=0, position=100)
    before = f.k.clone(), f.ks.clone()
    ops._rotor_encode(f.p[:, :128].reshape(3, 1, 128), f.k, f.ks, f.pos,
                       status=f.status, valid_rows=f.live)
    assert torch.equal(f.k, before[0]) and torch.equal(f.ks, before[1])
    assert f.status.tolist() == [0, -2]
    x = torch.full((1, 1, 128), 60000., dtype=torch.float32)
    ops._rotor_encode(x, f.k, f.ks, torch.tensor([0]), status=f.status, layer=7)
    assert f.status.tolist() == [8, 7]
    assert f.ks[0].item() == 0


@pytest.mark.parametrize("kb,kf,vb,vf", [
    (4, 4, 4, 4), (4, 4, 0, 0), (0, 0, 4, 4), (4, 4, 8, 0), (8, 0, 4, 4),
    (4, 5, 4, 5), (4, 5, 4, 4), (4, 4, 4, 5), (4, 5, 8, 0), (8, 0, 4, 5),
    (4, 5, 0, 0), (0, 0, 4, 5),
])
def test_stored_rotor_attention_matches_independent_q4_reference(kb, kf, vb, vf):
    torch.manual_seed(321)
    rows, keys, heads, width = 3, 67, 3, 128
    query = torch.randn(rows, heads, width).half()
    key = torch.randn(keys, 1, width).half()
    value = torch.randn_like(key)

    variants = {4: "signed-iso128", 5: "signed-iso64-norm"}

    def encode(x, bits, codec):
        if bits == 4:
            payload, scales = ref.quantize_ref(x, 4, variants[codec])
            return payload, scales, ref.dequant_ref(payload, scales, 4, variants[codec]).half()
        if bits == 8:
            payload, scales = ops.quantize_int8(x)
            return payload, scales, ops.dequantize_int8(payload, scales)
        return x, None, x

    k, ks, kd = encode(key, kb, kf)
    v, vs, vd = encode(value, vb, vf)
    q = (ref.rotate_ref(query, variants[kf]).half() if kb == 4 else
         ops.h32(query).half() if kb else query)
    scratch = SimpleNamespace(qsa=False, out=torch.empty_like(query))
    actual = ops.attention(q, k, v, torch.tensor([64]), scratch, rows, .125, context=keys,
                           ks=ks, vs=vs, k_bits=kb, k_codec=kf,
                           v_bits=vb, v_codec=vf, _native_allowed=False)
    expected = []
    for row in range(rows):
        end = 65 + row
        scores = q[row].float() @ kd[:end, 0].float().T * .125
        probability = scores.softmax(-1).half()
        result = probability.float() @ vd[:end, 0].float()
        if vb == 4:
            result = ref.rotate_ref(result, variants[vf], inverse=True)
        elif vb:
            result = ops.h32(result)
        expected.append(result.half())
    assert torch.equal(actual, torch.stack(expected))


@pytest.mark.parametrize("kb,kf,vb,vf", [
    (4, 5, 4, 5), (4, 5, 4, 4), (4, 4, 4, 5), (4, 5, 8, 0),
    (8, 0, 4, 5), (4, 5, 0, 0), (0, 0, 4, 5),
])
def test_norm_preserving_cpu_preparation_keeps_the_selected_side_metadata(kb, kf, vb, vf):
    f = fixture(kb, vb, kf=kf, vf=vf)
    prepare(f)
    assert f.status.tolist() == [0, -2]
    if vf:
        at = 2 * f.qh * f.dim + f.kh * f.dim
        source = f.p[:2, at:at + f.kh * f.dim].reshape(2, f.kh, f.dim)
        variant = "signed-iso64-norm" if vf == 5 else "signed-iso128"
        payload, scales = ref.quantize_ref(source, 4, variant)
        assert torch.equal(f.v[2:4], payload)
        assert torch.equal(f.vs[2:4], scales)


def test_existing_fp16_int8_native_preparation_keeps_its_original_argument_packet(monkeypatch):
    f = fixture(8, 8)
    calls = []
    monkeypatch.setattr(ops, "_fused", lambda _: SimpleNamespace(prep=lambda *a: calls.append(a)))
    prepare(f)
    assert len(calls) == 1 and len(calls[0]) == 27
    assert calls[0][21:24] == (8, 8, 5)


def test_q4_native_preparation_selects_new_entry_with_ordered_codecs(monkeypatch):
    f = fixture(4, 8)
    calls = []
    monkeypatch.setattr(ops, "_fused", lambda _: SimpleNamespace(prep_q4=lambda *a: calls.append(a)))
    prepare(f)
    assert len(calls) == 1 and len(calls[0]) == 29
    assert calls[0][-2:] == (4, 0)

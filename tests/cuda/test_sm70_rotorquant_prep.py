"""Fused Q4 prep boundaries against legacy raw prep and independent codec stages."""
from types import SimpleNamespace

import pytest
import torch

from tensorfold.cuda.sm70 import family_native, rotorquant_q4_native as codec
from tensorfold.families.qwen4_exp.cuda import rotorquant_ref as reference

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)
pytestmark = pytest.mark.skipif(torch.cuda.get_device_capability() != (7, 0), reason="SM70 fused Q4 prep")


PAIRS = ((4, 4, 4, 4), (4, 8, 4, 0), (8, 4, 0, 4), (4, 0, 4, 0), (0, 4, 0, 4),
         (4, 4, 5, 5), (4, 8, 5, 0), (8, 4, 0, 5), (4, 0, 5, 0), (0, 4, 0, 5),
         (4, 4, 4, 5), (4, 4, 5, 4))


def fixture(*, dim=256, rows=3, capacity=12, base=3, kbits=4, vbits=4, kcodec=None, vcodec=None):
    torch.manual_seed(713 + rows + dim)
    qh, kh, ih, iw = 3, 1, 4, 128
    projection = torch.randn((rows, 2 * qh * dim + 2 * kh * dim + (ih + 1) * iw),
                             device="cuda", dtype=torch.float16)
    weights = [torch.randn(size, device="cuda") * 0.1 + 1 for size in (dim, dim, iw)]
    frequency = torch.tensor([1e7 ** (-2 * i / 64) for i in range(32)], device="cuda")
    position = torch.tensor([base], device="cuda", dtype=torch.int64)
    delta = torch.tensor([2], device="cuda", dtype=torch.int32)
    live = torch.tensor([rows], device="cuda", dtype=torch.int32)
    status = torch.tensor([0, -2], device="cuda", dtype=torch.int32)

    def side(bits):
        payload = torch.full((capacity, kh, dim // 2 if bits == 4 else dim), 7,
                             device="cuda", dtype={0: torch.float16, 8: torch.int8, 4: torch.uint8}[bits])
        scales = (None if not bits else
                  torch.full((capacity, kh, dim // (128 if bits == 4 else 32)), 7,
                             device="cuda", dtype=torch.float32 if bits == 4 else torch.float16))
        return payload, scales

    key, ks = side(kbits)
    value, vs = side(vbits)
    query = torch.full((rows, qh, dim), 7, device="cuda", dtype=torch.float16)
    iq = torch.full((rows, ih, iw), 7, device="cuda", dtype=torch.float16)
    index = torch.full((capacity, iw), 7, device="cuda", dtype=torch.float16)
    kcodec = (4 if kbits == 4 else 0) if kcodec is None else kcodec
    vcodec = (4 if vbits == 4 else 0) if vcodec is None else vcodec
    args = [projection, position, *weights, frequency, query, key, value, iq, index,
            ks, vs, status, delta, live, qh, kh, dim, ih, iw, kbits, vbits, 9, 1e-6, 1.0, False,
            kcodec, vcodec]
    return SimpleNamespace(args=args, projection=projection, position=position, live=live, status=status,
                           query=query, key=key, value=value, iq=iq, index=index, ks=ks, vs=vs,
                           dim=dim, rows=rows, qh=qh, kh=kh, iw=iw, capacity=capacity, kcodec=kcodec, vcodec=vcodec)


def invoke(case):
    family_native.extension().prep_q4(*case.args)


def raw_reference(case):
    # The previously qualified raw preparation supplies unchanged norm/RoPE/Half
    # boundaries. The independent Q4 table/rotation oracle is applied afterward.
    q = torch.empty_like(case.query)
    k = torch.zeros((case.capacity, case.kh, case.dim), device="cuda", dtype=torch.float16)
    v = torch.zeros_like(k)
    iq = torch.empty_like(case.iq)
    ik = torch.zeros_like(case.index)
    args = list(case.args[:-2])
    args[6:11] = [q, k, v, iq, ik]
    args[11:14] = [None, None, torch.tensor([0, -2], device="cuda", dtype=torch.int32)]
    args[21:23] = [0, 0]
    family_native.extension().prep(*args)
    return q, k, v, iq, ik


@pytest.mark.parametrize("dim", [128, 256])
@pytest.mark.parametrize("rows", [1, 3, 17])
@pytest.mark.parametrize("kb,vb,kcodec,vcodec", PAIRS)
def test_q4_fusion_preserves_raw_preparation_boundaries_and_codec_stages(dim, rows, kb, vb, kcodec, vcodec):
    case = fixture(dim=dim, rows=rows, capacity=rows + 8, kbits=kb, vbits=vb, kcodec=kcodec, vcodec=vcodec)
    raw = raw_reference(case)
    invoke(case)
    assert case.status.tolist() == [0, -2]
    variant = "signed-iso64-norm" if kcodec == 5 else "signed-iso128"
    expected_query = (reference.rotate_ref(raw[0], variant).half() if kb == 4 else raw[0])
    if kb == 8:
        from tensorfold.cuda.sm70.family_ops import h32
        expected_query = h32(expected_query).half()
    torch.testing.assert_close(case.query, expected_query, rtol=0, atol=0)
    torch.testing.assert_close(case.iq, raw[3], rtol=0, atol=0)
    torch.testing.assert_close(case.index[3:3 + rows], raw[4][3:3 + rows], rtol=0, atol=0)
    for bits, selected_codec, actual, scales, original in (
            (kb, kcodec, case.key, case.ks, raw[1]), (vb, vcodec, case.value, case.vs, raw[2])):
        if bits == 4:
            codes = torch.full_like(actual, 7)
            rms = torch.full_like(scales, 7)
            codec.encode(original[3:3 + rows], codes, rms, position=case.position, codec=selected_codec)
            torch.testing.assert_close(actual, codes, rtol=0, atol=0)
            torch.testing.assert_close(scales, rms, rtol=0, atol=0)
        elif bits == 8:
            from tensorfold.cuda.sm70.family_ops import quantize_int8
            codes, rms = quantize_int8(original[3:3 + rows])
            torch.testing.assert_close(actual[3:3 + rows], codes, rtol=0, atol=0)
            torch.testing.assert_close(scales[3:3 + rows], rms, rtol=0, atol=0)
        else:
            torch.testing.assert_close(actual[3:3 + rows], original[3:3 + rows], rtol=0, atol=0)
        assert (actual[:3] == 7).all() and (actual[3 + rows:] == 7).all()


@pytest.mark.parametrize("selected_codec", [4, 5])
def test_padded_nan_rows_cannot_poison_or_write_other_cache_rows(selected_codec):
    case = fixture(kcodec=selected_codec, vcodec=selected_codec)
    case.live.fill_(2)
    case.projection[2].fill_(float("nan"))
    invoke(case)
    assert case.status.tolist() == [0, -2]
    for payload in (case.key, case.value, case.ks, case.vs, case.index):
        assert (payload[:3] == 7).all() and (payload[5:] == 7).all()
    assert (case.query[2] == 7).all() and (case.iq[2] == 7).all()


@pytest.mark.parametrize("selected_codec", [4, 5])
def test_captured_dynamic_position_live_count_and_first_error_stay_device_owned(selected_codec):
    case = fixture(capacity=16, base=1, kcodec=selected_codec, vcodec=selected_codec)
    invoke(case)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        invoke(case)
    case.position.fill_(7)
    case.live.fill_(1)
    case.projection[1:].fill_(float("nan"))
    graph.replay()
    assert case.status.tolist() == [0, -2]
    assert (case.key[8:] == 7).all() and (case.value[8:] == 7).all()
    case.projection[0, 0] = float("nan")
    graph.replay()
    assert case.status.tolist() == [1, 9]
    case.status[1] = 5
    graph.replay()
    assert case.status.tolist() == [1, 5]


@pytest.mark.parametrize("change", ["negative_position", "huge_position", "live_negative", "live_too_large"])
def test_invalid_device_controls_fail_before_source_or_cache_access(change):
    case = fixture()
    if change == "negative_position":
        case.position.fill_(-1)
    elif change == "huge_position":
        case.position.fill_(2**63 - 1)
    else:
        case.live.fill_(-1 if change == "live_negative" else case.rows + 1)
    case.projection.fill_(float("nan"))
    invoke(case)
    assert case.status.tolist() == [16, 9]
    assert all((tensor == 7).all() for tensor in (case.key, case.value, case.ks, case.vs, case.index))


@pytest.mark.parametrize("selected_codec", [4, 5])
def test_finite_wire_source_that_overflows_half_reconstruction_is_contained(selected_codec):
    case = fixture(kbits=0, vbits=4, vcodec=selected_codec)
    values_at = 2 * case.qh * case.dim + case.kh * case.dim
    case.projection[:, values_at:values_at + case.dim].fill_(65504)
    invoke(case)
    assert case.status.tolist() == [8, 9]
    assert (case.value[3:6] == 0x77).all() and (case.vs[3:6] == 0).all()


@pytest.mark.parametrize("change", ["payload", "scale_dtype", "scale_shape", "unsupported_codec", "bits_codec", "position",
                                    "epsilon_range", "epsilon_underflow", "rotary_range", "rotary_underflow", "layer_sentinel"])
def test_structural_contracts_reject_before_launch(change):
    case = fixture()
    if change == "payload":
        case.args[7] = case.key.to(torch.int8)
    elif change == "scale_dtype":
        case.args[11] = case.ks.half()
    elif change == "scale_shape":
        case.args[11] = case.ks.reshape(-1)
    elif change == "unsupported_codec":
        case.args[-2] = 6
    elif change == "bits_codec":
        case.args[21] = 8
    elif change == "position":
        case.args[1] = case.position.float()
    elif change == "layer_sentinel":
        case.args[23] = -2
    else:
        at = 24 if change.startswith("epsilon") else 25
        case.args[at] = 1e300 if change.endswith("range") else 1e-300
    with pytest.raises(RuntimeError):
        invoke(case)


@pytest.mark.parametrize("bits,selected_codec", [(4, 1), (4, 2), (4, 3), (4, 6), (8, 5), (0, 5)])
def test_other_rotor_variants_or_equal_width_impostors_remain_rejected(bits, selected_codec):
    case = fixture()
    case.args[21] = bits
    case.args[-2] = selected_codec
    with pytest.raises(RuntimeError, match="CODEC4-or-5"):
        invoke(case)

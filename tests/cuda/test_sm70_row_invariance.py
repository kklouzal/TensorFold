"""Actual TP8 attention/GDN stored prefixes equal serial logical execution."""
from unittest.mock import patch

import pytest
import torch

from tensorfold.cuda.sm70 import family_ops as ops
from tensorfold.families.qwen4_exp.cuda import gdn, glue

pytestmark = pytest.mark.skipif(torch.cuda.get_device_capability() != (7, 0), reason="SM70 row invariance")


def _bits(actual, expected, field):
    dtype = {torch.float16: torch.int16, torch.float32: torch.int32}.get(actual.dtype, actual.dtype)
    torch.testing.assert_close(actual.contiguous().view(dtype), expected.contiguous().view(dtype), atol=0, rtol=0,
                               msg=f"row-independent {field}")


def prep_inputs(rows, base, *, qheads=3, kvheads=1, seed=1237):
    torch.manual_seed(seed)
    dim, ih, iw = 256, 4, 128
    p = torch.randn((rows, 2 * qheads * dim + 2 * kvheads * dim + (ih + 1) * iw), device="cuda", dtype=torch.float16) * 3
    weights = [torch.randn(size, device="cuda") * .2 + 1 for size in (dim, dim, iw)]
    freq = torch.tensor([1e7 ** (-2 * i / 64) for i in range(32)], device="cuda")
    return p, weights, freq, base, qheads, kvheads


def prepare(source, kb, vb, *, serial=False, native=True, independent_rows=False, controls=None):
    p, weights, freq, base, qheads, kvheads = source
    rows, dim, ih, iw = p.shape[0], 256, 4, 128
    q = torch.empty((rows, qheads, dim), device="cuda", dtype=torch.float16)
    iq = torch.empty((rows, ih, iw), device="cuda", dtype=torch.float16)
    k = torch.zeros((base + rows, kvheads, dim), device="cuda", dtype=torch.int8 if kb else torch.float16)
    v = torch.zeros_like(k, dtype=torch.int8 if vb else torch.float16)
    ks, vs = [torch.zeros((base + rows, kvheads, dim // 32), device="cuda", dtype=torch.float16) for _ in range(2)]
    index = torch.zeros((base + rows, iw), device="cuda", dtype=torch.float16)
    status = controls[0] if controls is not None else torch.tensor([0, -2], device="cuda", dtype=torch.int32)
    with patch.object(ops, "_NATIVE_FAMILY", native):
        for at, count in ([(i, 1) for i in range(rows)] if serial else [(0, rows)]):
            pos = controls[1] if controls is not None else torch.tensor([base + at], device="cuda", dtype=torch.int32)
            section = slice(at, at + count)
            glue.attn_prep(p[section], pos, *weights, freq, q[section], k, v, iq[section], index, 1e-6,
                           q_heads=qheads, kv_heads=kvheads, head_dim=dim, index_heads=ih, index_dim=iw,
                           ks=ks, vs=vs, k_bits=kb, k_codec=0, v_bits=vb, v_codec=0, status=status,
                           independent_rows=independent_rows)
    return dict(q=q, iq=iq, k=k[base:], v=v[base:], ks=ks[base:], vs=vs[base:], ikc=index[base:], status=status)


@pytest.mark.parametrize("rows", [2, 3])
@pytest.mark.parametrize("base", [27, 128, 8192])
@pytest.mark.parametrize("kb,vb", [(0, 0), (8, 8), (0, 8), (8, 0)])
@pytest.mark.parametrize("native", [False, True])
def test_actual_tp8_attention_prep_windows_match_all_serial_stored_bytes(rows, base, kb, vb, native):
    source = prep_inputs(rows, base)
    serial, window = prepare(source, kb, vb, serial=True, native=native), prepare(source, kb, vb, native=native)
    for name in serial:
        _bits(window[name], serial[name], name)


@pytest.mark.parametrize("rows", [2, 3])
@pytest.mark.parametrize("seed", [113, 317, 1002])
@pytest.mark.parametrize("amplitude", [.2, 3.0])
def test_actual_tp8_gdn_all_saved_core_bytes_outputs_and_states_match_serial(rows, seed, amplitude):
    torch.manual_seed(seed)
    nk, nv, dk, dv = 2, 6, 128, 128
    channels, width = gdn.widths(nk, nv)
    p = torch.randn((rows, width), device="cuda", dtype=torch.float16) * amplitude
    conv = torch.randn((3, channels), device="cuda", dtype=torch.float16) * .2
    weight = torch.randn((channels, 4), device="cuda", dtype=torch.float16) * .2
    initial = torch.randn((nv, dv, dk), device="cuda") * .01
    log, bias = [torch.randn(nv, device="cuda") * .1 for _ in range(2)]
    norm = torch.randn(dv, device="cuda", dtype=torch.float16) * .2 + 1
    scratch = gdn.GDNScratch(rows, "cuda", nk, nv, dtype=torch.float16)
    out = torch.empty((rows, nv * dv), device="cuda", dtype=torch.float16)
    sums = torch.empty((rows, nv * dv // 32), device="cuda")
    final = torch.empty_like(initial)
    gdn.chain(p, conv, weight, initial, log, bias, norm, 1e-6, rows, scratch, final, out, sums)
    state, tail = initial.clone(), conv.clone()
    for row in range(rows):
        single = gdn.GDNScratch(1, "cuda", nk, nv, dtype=torch.float16)
        one, one_sums = torch.empty_like(out[:1]), torch.empty_like(sums[:1])
        next_state = torch.empty_like(state)
        gdn.chain(p[row:row + 1], tail, weight, state, log, bias, norm, 1e-6, 1, single, next_state, one, one_sums)
        for name in ("k", "v", "g", "b"):
            _bits(getattr(scratch, name)[row:row + 1], getattr(single, name), name)
        _bits(out[row:row + 1], one, "gdn output")
        _bits(sums[row:row + 1], one_sums, "gdn group sums")
        replay = torch.empty_like(state)
        gdn.replay(initial, scratch, row + 1, replay)
        _bits(replay, next_state, "accepted-prefix state")
        state = next_state
        ops.shift_windows(tail[None], p[None, row:row + 1], 1, channels)
    _bits(final, state, "final recurrent state")


@pytest.mark.parametrize("rows", [6, 8, 18])
@pytest.mark.parametrize("base", [27, 128, 8192])
@pytest.mark.parametrize("kb,vb", [(0, 0), (8, 8), (0, 8), (8, 0)])
@pytest.mark.parametrize("native", [False, True])
def test_deep_tp8_decode_prep_keeps_all_serial_rows_exact(rows, base, kb, vb, native):
    source = prep_inputs(rows, base)
    serial = prepare(source, kb, vb, serial=True, native=native)
    decoded = prepare(source, kb, vb, native=native, independent_rows=True)
    for name in serial:
        _bits(decoded[name], serial[name], "deep " + name)


@pytest.mark.parametrize("rows", [6, 8, 18])
@pytest.mark.parametrize("kb,vb", [(0, 0), (8, 8), (0, 8), (8, 0)])
def test_captured_deep_decode_prep_replays_dynamic_projection_rows_exactly(rows, kb, vb):
    source = prep_inputs(rows, 128)
    controls = (torch.tensor([0, -2], device="cuda", dtype=torch.int32),
                torch.tensor([128], device="cuda", dtype=torch.int32))
    prepare(source, kb, vb, independent_rows=True, controls=controls)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = prepare(source, kb, vb, independent_rows=True, controls=controls)
    for multiplier in (.5, 2., -1.):
        source[0].mul_(multiplier)
        serial = prepare(source, kb, vb, serial=True)
        graph.replay()
        for name in serial:
            _bits(captured[name], serial[name], "captured deep " + name)


@pytest.mark.parametrize("rows", [6, 8, 18])
@pytest.mark.parametrize("kb,vb", [(0, 0), (8, 8), (0, 8), (8, 0)])
def test_prompt_prep_keeps_original_batch_geometry_and_codec_bytes(rows, kb, vb):
    source = prep_inputs(rows, 128)
    reference, native = prepare(source, kb, vb, native=False), prepare(source, kb, vb)
    for name in reference:
        _bits(native[name], reference[name], "unchanged prompt " + name)

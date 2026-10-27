"""Fused family candidates against retained Torch oracles at this host's geometry."""

from unittest.mock import patch

import pytest
import torch

from tensorfold.cuda.sm70 import family_ops as ops
from tensorfold.families.qwen4_exp.cuda import gdn, glue

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0),
                               reason="SM70 fused family")


def invoke(name, native, *args, **kwargs):
    with patch.object(ops, "_NATIVE_FAMILY", native):
        return getattr(ops, name)(*args, **kwargs)


@pytest.mark.parametrize("mode", range(5))
def test_hyperconnection_host_geometry_writeback_norm_mix_against_torch(mode):
    torch.manual_seed(190 + mode)
    rows, streams, dims = 3, 4, 2560
    h = torch.randn((rows, streams * dims), device="cuda", dtype=torch.float16) * 0.1
    inject = torch.randn((rows, streams), device="cuda", dtype=torch.float16) * 0.1
    branch = (torch.randn((rows, dims), device="cuda", dtype=torch.float16) if mode == 1 else
              torch.randn((8, rows, dims), device="cuda")) * 0.1
    y = torch.randn((rows, 11, dims), device="cuda") * 0.1
    weights = torch.randn((rows, 11), device="cuda").softmax(-1)
    scale = torch.randn(streams * dims, device="cuda") * 0.1 + 1
    up = torch.randn_like(h) * 0.1
    results = []
    for native in (False, True):
        out = h.clone()
        pss = torch.empty((rows, dims // 256, streams), device="cuda")
        invoke("hc_writeback", native, out, out, pss, streams, mode,
               branch=branch, inject=inject, y=y, wts=weights)
        normed, xs = torch.empty_like(out), torch.empty((rows, streams * dims // 32), device="cuda")
        invoke("hc_normed", native, out, pss, scale, normed, xs, streams, 1e-6)
        mixed, sums = torch.empty((rows, dims), device="cuda", dtype=torch.float16), torch.empty((rows, dims // 32), device="cuda")
        invoke("hc_mix", native, up, normed, mixed, sums, streams)
        results.append((out, pss, normed, xs, mixed, sums))
    torch.testing.assert_close(results[1][0], results[0][0], atol=0, rtol=0)
    for index in (1, 3, 5):
        torch.testing.assert_close(results[1][index], results[0][index], atol=0.01, rtol=0.003)
    for index in (2, 4):
        torch.testing.assert_close(results[1][index], results[0][index], atol=0.002, rtol=0.002)


@pytest.mark.parametrize("splits", [0, 1, 8])
def test_down_activation_injection_preserves_half_boundaries(splits):
    torch.manual_seed(83)
    rows, low, streams = 3, 320, 4
    source = torch.randn(((splits, rows, low + streams) if splits else (rows, low + streams)), device="cuda")
    results = []
    for native in (False, True):
        act = torch.empty((rows, low), device="cuda", dtype=torch.float16)
        xs, inject = torch.empty((rows, low // 32), device="cuda"), torch.empty((rows, streams), device="cuda", dtype=torch.float16)
        invoke("hc_reduce_act" if splits else "hc_act", native, source, act, xs, inject, streams, low)
        results.append((act, xs, inject))
    torch.testing.assert_close(results[1][0], results[0][0], atol=0.0005, rtol=0.001)
    torch.testing.assert_close(results[1][1], results[0][1], atol=0.002, rtol=0.001)
    torch.testing.assert_close(results[1][2], results[0][2], atol=0, rtol=0)


def test_native_silu_matches_torch_for_every_finite_fp16_input():
    values = torch.arange(65536, device="cuda", dtype=torch.int32).to(torch.int16).view(torch.float16).reshape(-1, 32)
    results = []
    for native in (False, True):
        out = torch.empty_like(values)
        sums = torch.empty((values.shape[0], 1), device="cuda")
        invoke("hc_act", native, values, out, sums, None, 1, 32)
        results.append(out)
    finite = values.isfinite()
    torch.testing.assert_close(results[1][finite], results[0][finite], atol=0, rtol=0)


@pytest.mark.parametrize("rows", [1, 3, 17])
def test_host_hc_pss_and_normalized_stored_values_match_torch_exactly(rows):
    torch.manual_seed(82)
    streams, dims = 4, 2560
    h = torch.randn((rows, streams * dims), device="cuda", dtype=torch.float16) * 3
    branch = torch.randn((8, rows, dims), device="cuda") * 0.3
    inject = torch.randn((rows, streams), device="cuda", dtype=torch.float16) * 0.3
    scale = torch.randn(streams * dims, device="cuda") * 0.2 + 1
    results = []
    for native in (False, True):
        out = h.clone()
        pss = torch.empty((rows, dims // 256, streams), device="cuda")
        normalized, sums = torch.empty_like(out), torch.empty((rows, streams * dims // 32), device="cuda")
        invoke("hc_writeback", native, out, out, pss, streams, 3, branch=branch, inject=inject)
        invoke("hc_normed", native, out, pss, scale, normalized, sums, streams, 1e-6)
        results.append((out, pss, normalized, sums))
    for actual, expected in zip(results[1], results[0]):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize("group", [2560, 10240])
def test_wide_rmsnorm_group_sums_against_torch(group):
    torch.manual_seed(7)
    value = torch.randn((3, 10240), device="cuda", dtype=torch.float16)
    weight = torch.randn(10240, device="cuda") * 0.1 + 1
    expected = invoke("rmsnorm", False, value, weight, 1e-6, group=group)
    actual = invoke("rmsnorm", True, value, weight, 1e-6, group=group)
    torch.testing.assert_close(actual[0], expected[0], atol=0.002, rtol=0.002)
    torch.testing.assert_close(actual[1], expected[1], atol=0.005, rtol=0.002)


def test_ple_host_geometry_gate_convolution_and_strided_commits():
    torch.manual_seed(77)
    rows, streams, dims = 3, 4, 2560
    h = torch.randn((rows, streams * dims), device="cuda", dtype=torch.float16) * 0.1
    keys, values = torch.randn_like(h) * 0.1, torch.randn((rows, dims), device="cuda", dtype=torch.float16) * 0.1
    weight = torch.ones(streams * dims, device="cuda")
    tail = torch.randn((9, streams * dims), device="cuda", dtype=torch.float16) * 0.1
    conv = torch.randn((streams * dims, 4), device="cuda", dtype=torch.float16) * 0.1
    results = []
    for native in (False, True):
        gated, pss = torch.empty_like(h), torch.empty((rows, streams), device="cuda")
        invoke("ple_gate", native, keys, values, h, weight, weight, gated, pss, 1e-6, streams)
        output, normalized = h.clone(), torch.empty_like(h)
        invoke("ple_conv", native, gated, pss, weight, tail, conv, output, output, normalized, 1e-6, streams, 3)
        results.append((gated, pss, output, normalized))
    for actual, expected in zip(results[1], results[0]):
        torch.testing.assert_close(actual, expected, atol=0.002, rtol=0.003)
    old = torch.randn((3, 9, 1280), device="cuda", dtype=torch.float16)
    new = torch.randn((3, 8, 2060), device="cuda", dtype=torch.float16)[:, 2:5]
    reference, actual = old.clone(), old.clone()
    invoke("shift_windows", False, reference, new, 2, 1280)
    invoke("shift_windows", True, actual, new, 2, 1280)
    torch.testing.assert_close(actual, reference, atol=0, rtol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_moe_partials_retain_fp32_slot_order(dtype):
    torch.manual_seed(73)
    y = torch.randn((3, 11, 2560), device="cuda", dtype=dtype)
    weights = torch.randn((3, 11), device="cuda").softmax(-1)
    expected, actual = [torch.empty((3, 2560), device="cuda") for _ in range(2)]
    invoke("moe_partial", False, y, weights, expected, 3)
    invoke("moe_partial", True, y, weights, actual, 3)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_gdn_captured_live_prefix_publishes_exact_state_for_padded_rows():
    torch.manual_seed(33)
    channels, width = gdn.widths(2, 6)
    p = torch.randn((3, width), device="cuda", dtype=torch.float16) * 0.1
    conv = torch.randn((3, channels), device="cuda", dtype=torch.float16) * 0.1
    weight = torch.randn((channels, 4), device="cuda", dtype=torch.float16) * 0.1
    initial = torch.randn((6, 128, 128), device="cuda") * 0.01
    log, bias, norm = torch.zeros(6, device="cuda"), torch.zeros(6, device="cuda"), torch.ones(128, device="cuda", dtype=torch.float16)
    scratch = gdn.GDNScratch(3, "cuda", 2, 6, dtype=torch.float16)
    scratch.keep_rows = torch.tensor([3], device="cuda", dtype=torch.int32)
    output, sums, final = torch.empty((3, 768), device="cuda", dtype=torch.float16), torch.empty((3, 24), device="cuda"), torch.empty_like(initial)

    def run():
        gdn.chain(p, conv, weight, initial, log, bias, norm, 1e-6, 3, scratch, final, output, sums)

    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    replayed = torch.empty_like(initial)
    for kept in (0, 1, 2, 3):
        scratch.keep_rows.fill_(kept)
        graph.replay()
        gdn.replay(initial, scratch, kept, replayed)
        torch.testing.assert_close(final, replayed, atol=0, rtol=0)
    # A nonfinite padded future row must not alter the live-prefix recurrent state.
    scratch.keep_rows.fill_(2)
    p[2].fill_(float("nan"))
    graph.replay()
    gdn.replay(initial, scratch, 2, replayed)
    torch.testing.assert_close(final, replayed, atol=0, rtol=0)


@pytest.mark.parametrize("native", [False, True])
def test_padded_attention_rows_do_not_poison_live_int8_frames(native):
    torch.manual_seed(11)
    rows, heads, kv, dim, ih, iw = 3, 3, 1, 256, 4, 128
    p = torch.randn((rows, heads * 2 * dim + 2 * kv * dim + (ih + 1) * iw), device="cuda", dtype=torch.float16)
    p[2].fill_(float("nan"))
    pos = torch.tensor([0], device="cuda", dtype=torch.int32)
    live = torch.tensor([2], device="cuda", dtype=torch.int32)
    status = torch.tensor([0, -2], device="cuda", dtype=torch.int32)
    q = torch.empty((rows, heads, dim), device="cuda", dtype=torch.float16)
    iq = torch.empty((rows, ih, iw), device="cuda", dtype=torch.float16)
    kc, vc = [torch.zeros((8, kv, dim), device="cuda", dtype=torch.int8) for _ in range(2)]
    ks, vs = [torch.zeros((8, kv, dim // 32), device="cuda", dtype=torch.float16) for _ in range(2)]
    ikc = torch.zeros((8, iw), device="cuda", dtype=torch.float16)
    scale, iscale = torch.ones(dim, device="cuda"), torch.ones(iw, device="cuda")
    frequency = torch.ones(32, device="cuda")

    def run():
        with patch.object(ops, "_NATIVE_FAMILY", native):
            glue.attn_prep(p, pos, scale, scale, iscale, frequency, q, kc, vc, iq, ikc, 1e-6,
                           q_heads=heads, kv_heads=kv, head_dim=dim, index_heads=ih, index_dim=iw,
                           ks=ks, vs=vs, bits=8, status=status, layer=9, valid_rows=live)

    run()
    assert status.tolist() == [0, -2]
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    live.fill_(3)
    graph.replay()
    assert status[0] != 0 and status[1] == 9


@pytest.mark.parametrize("kb,vb", [(0, 0), (8, 8), (0, 8), (8, 0)])
def test_native_attention_prep_host_geometry_and_existing_int8_bytes(kb, vb):
    torch.manual_seed(317)
    rows, qh, kh, hd, ih, iw, capacity = 3, 3, 1, 256, 4, 128, 12
    packed = torch.randn((rows, 2 * qh * hd + 2 * kh * hd + (ih + 1) * iw), device="cuda", dtype=torch.float16)
    position = torch.tensor([3], device="cuda", dtype=torch.int32)
    delta = torch.tensor([2], device="cuda", dtype=torch.int32)
    qw, kw, iqw = [torch.randn(size, device="cuda") * 0.1 + 1 for size in (hd, hd, iw)]
    frequency = torch.tensor([1e7 ** (-2 * i / 64) for i in range(32)], device="cuda")

    def prepare(native, key_bits, value_bits):
        query = torch.zeros((rows, qh, hd), device="cuda", dtype=torch.float16)
        key = torch.zeros((capacity, kh, hd), device="cuda", dtype=torch.int8 if key_bits else torch.float16)
        value = torch.zeros((capacity, kh, hd), device="cuda", dtype=torch.int8 if value_bits else torch.float16)
        kscale, vscale = [torch.zeros((capacity, kh, hd // 32), device="cuda", dtype=torch.float16) for _ in range(2)]
        iq = torch.zeros((rows, ih, iw), device="cuda", dtype=torch.float16)
        index = torch.zeros((capacity, iw), device="cuda", dtype=torch.float16)
        status = torch.tensor([0, -2], device="cuda", dtype=torch.int32)
        invoke("attn_prep", native, packed, position, qw, kw, iqw, frequency, query, key, value, iq, index, 1e-6,
               q_heads=qh, kv_heads=kh, head_dim=hd, index_heads=ih, index_dim=iw, ks=kscale, vs=vscale,
               k_bits=key_bits, k_codec=0, v_bits=value_bits, v_codec=0, delta=delta, status=status)
        assert status.tolist() == [0, -2]
        return query, key, value, iq, index, kscale, vscale

    reference, actual, raw = prepare(False, kb, vb), prepare(True, kb, vb), prepare(True, 0, 0)
    torch.testing.assert_close(actual[0], reference[0], atol=0.004, rtol=0.003)
    torch.testing.assert_close(actual[3], reference[3], atol=0.004, rtol=0.003)
    torch.testing.assert_close(actual[4], reference[4], atol=0, rtol=0)
    for index, scale_index, bits in ((1, 5, kb), (2, 6, vb)):
        if bits:
            encoded, scales = ops.quantize_int8(raw[index][3:6])
            torch.testing.assert_close(actual[index][3:6], encoded, atol=0, rtol=0)
            torch.testing.assert_close(actual[scale_index][3:6], scales, atol=0, rtol=0)
        else:
            torch.testing.assert_close(actual[index], reference[index], atol=0.004, rtol=0.003)


@pytest.mark.parametrize("rows", [1, 2, 3, 9, 17, 37])
@pytest.mark.parametrize("bits", [0, 8])
@pytest.mark.parametrize("independent_rows", [False, True])
def test_prep_exact_torch_stored_bytes_across_prefill_and_decode_reduction_shapes(rows, bits, independent_rows):
    torch.manual_seed(1237)
    qh, kh, hd, ih, iw, base = 3, 1, 256, 4, 128, 273
    packed = torch.randn((rows, 2 * qh * hd + 2 * kh * hd + (ih + 1) * iw), device="cuda", dtype=torch.float16) * 3
    position = torch.tensor([base], device="cuda", dtype=torch.int32)
    qw, kw, iqw = [torch.randn(size, device="cuda") * 0.2 + 1 for size in (hd, hd, iw)]
    frequency = torch.tensor([1e7 ** (-2 * i / 64) for i in range(32)], device="cuda")
    results = []
    for native in (False, True):
        q = torch.empty((rows, qh, hd), device="cuda", dtype=torch.float16)
        iq = torch.empty((rows, ih, iw), device="cuda", dtype=torch.float16)
        key, value = [torch.zeros((base + rows, kh, hd), device="cuda", dtype=torch.int8 if bits else torch.float16) for _ in range(2)]
        ks, vs = [torch.zeros((base + rows, kh, hd // 32), device="cuda", dtype=torch.float16) for _ in range(2)]
        index = torch.zeros((base + rows, iw), device="cuda", dtype=torch.float16)
        status = torch.tensor([0, -2], device="cuda", dtype=torch.int32)
        invoke("attn_prep", native, packed, position, qw, kw, iqw, frequency, q, key, value, iq, index, 1e-6,
               q_heads=qh, kv_heads=kh, head_dim=hd, index_heads=ih, index_dim=iw,
               ks=ks, vs=vs, bits=bits, status=status, independent_rows=independent_rows)
        results.append((q, key, value, iq, index, ks, vs))
    for actual, expected in zip(results[1], results[0]):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize("rows", [1, 2, 3, 5, 17])
def test_ple_exact_torch_gate_pss_and_convolution_bytes(rows):
    torch.manual_seed(493)
    streams, dims = 4, 2560
    h = torch.randn((rows, streams * dims), device="cuda", dtype=torch.float16) * 3
    keys = torch.randn_like(h) * 3
    values = torch.randn((rows, dims), device="cuda", dtype=torch.float16) * 0.7
    nk, nq = [torch.randn(streams * dims, device="cuda") * 0.2 + 1 for _ in range(2)]
    tail = torch.randn((9, streams * dims), device="cuda", dtype=torch.float16) * 0.4
    conv = torch.randn((streams * dims, 4), device="cuda", dtype=torch.float16) * 0.4
    norm = torch.randn(streams * dims, device="cuda") * 0.2 + 1
    results = []
    for native in (False, True):
        gated = torch.empty_like(h)
        pss = torch.empty((rows, streams), device="cuda")
        invoke("ple_gate", native, keys, values, h, nk, nq, gated, pss, 1e-6, streams)
        output, normalized = torch.empty_like(h), torch.empty_like(h)
        invoke("ple_conv", native, gated, pss, norm, tail, conv, h, output, normalized, 1e-6, streams, 3)
        results.append((gated, pss, output, normalized))
    for actual, expected in zip(results[1], results[0]):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize("kept", [1, 2, 3])
def test_padded_ple_uses_real_global_row_count_for_exact_torch_reduction(kept):
    torch.manual_seed(42)
    rows, streams, dims = 3, 4, 2560
    h = torch.randn((rows, streams * dims), device="cuda", dtype=torch.float16)
    keys = torch.randn_like(h)
    values = torch.randn((rows, dims), device="cuda", dtype=torch.float16)
    scale = torch.randn(streams * dims, device="cuda") * 0.2 + 1
    expected, actual = torch.empty_like(h[:kept]), torch.empty_like(h)
    epss, apss = torch.empty((kept, streams), device="cuda"), torch.empty((rows, streams), device="cuda")
    invoke("ple_gate", False, keys[:kept], values[:kept], h[:kept], scale, scale, expected, epss, 1e-6, streams)
    live = torch.tensor([kept], device="cuda", dtype=torch.int32)
    invoke("ple_gate", True, keys, values, h, scale, scale, actual, apss, 1e-6, streams, valid_rows=live)
    torch.testing.assert_close(actual[:kept], expected, atol=0, rtol=0)
    torch.testing.assert_close(apss[:kept], epss, atol=0, rtol=0)


def test_full_family_native_logits_and_recurrent_state_against_golden_torch():
    from test_flashnext_sm70_family import _small_weights
    from tensorfold.families.qwen4_exp.cuda.forward import commit, forward
    from tensorfold.families.qwen4_exp.cuda.state import Buffers, State

    w = _small_weights(True)
    tokens = [4, 17, 9, 31, 16, 25, 13, 15, 7]
    golden = []
    for native in (False, True):
        state, buf = State(w, 24, 12, "fp16"), Buffers(w, 12, 24)
        with patch.object(ops, "_NATIVE_FAMILY", native):
            logits = forward(w, state, buf, tokens).clone()
            commit(w, state, buf, len(tokens), len(tokens))
        golden.append((logits, state.rec[state.cur[0]].clone(), state.conv.clone(), state.ple_tail.clone()))
    torch.testing.assert_close(golden[1][0], golden[0][0], atol=0.01, rtol=0.005)
    torch.testing.assert_close(golden[1][0].argmax(-1), golden[0][0].argmax(-1), atol=0, rtol=0)
    torch.testing.assert_close(golden[1][1], golden[0][1], atol=2e-5, rtol=0.02)
    for index in (2, 3):
        torch.testing.assert_close(golden[1][index], golden[0][index], atol=0.002, rtol=0.005)


@pytest.mark.parametrize("blocks", [1024, 2048, 4096])
@pytest.mark.parametrize("pattern", ["random", "zero", "cutoff"])
def test_native_sparse_selector_stable_ids_and_scores_against_torch(blocks, pattern):
    from tensorfold.families.qwen4_exp.cuda.attention import AttnScratch

    torch.manual_seed(151)
    rows, ratio, budget, dim = 3, 4, 2048, 128
    queries = torch.randn((rows, 4, dim), device="cuda", dtype=torch.float16)
    pooled = torch.randn((blocks, dim), device="cuda", dtype=torch.float16)
    if pattern != "random":
        queries.zero_()
        pooled.zero_()
        pooled[::2, 0] = -0.0
        queries[:, :, 0] = 1
        if pattern == "cutoff":
            pooled[:, 0] = (torch.arange(blocks, device="cuda") % 5 - 2).half()
    position = torch.tensor([blocks * ratio - 9], device="cuda", dtype=torch.int32)
    reference, native = [AttnScratch(rows, 3, 256, blocks * ratio, "cuda", budget=budget,
                                    ratio=ratio, dtype=torch.float16) for _ in range(2)]
    for scratch, enabled in ((reference, False), (native, True)):
        invoke("qsa_rows", enabled, queries, pooled, position, scratch, rows, context=blocks * ratio)
    for field in ("sparse", "nk", "ids"):
        torch.testing.assert_close(getattr(native, field), getattr(reference, field), atol=0, rtol=0)
    torch.testing.assert_close(native.scores[:, :blocks], reference.scores[:, :blocks], atol=2e-5, rtol=2e-5)


def test_native_qsa_graph_crosses_dense_sparse_boundary_with_one_context_bucket():
    from tensorfold.families.qwen4_exp.cuda.attention import AttnScratch

    torch.manual_seed(42)
    rows, ratio, budget, dim, blocks = 3, 4, 2048, 128, 1024
    queries = torch.zeros((rows, 4, dim), device="cuda", dtype=torch.float16)
    pooled = torch.zeros((blocks, dim), device="cuda", dtype=torch.float16)
    position = torch.tensor([2049], device="cuda", dtype=torch.int32)
    reference, native = [AttnScratch(rows, 3, 256, blocks * ratio, "cuda", budget=budget,
                                    ratio=ratio, dtype=torch.float16) for _ in range(2)]

    def run():
        invoke("qsa_rows", True, queries, pooled, position, native, rows, context=blocks * ratio)

    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for start in (2049, 2052, 2055):
        position.fill_(start)
        graph.replay()
        invoke("qsa_rows", False, queries, pooled, position, reference, rows, context=blocks * ratio)
        for row in range(rows):
            assert native.nk[row] == reference.nk[row] and native.sparse[row] == reference.sparse[row]
            if bool(native.sparse[row]):
                torch.testing.assert_close(native.ids[row], reference.ids[row], atol=0, rtol=0)


@pytest.mark.parametrize("kept", [1, 2])
@pytest.mark.parametrize("mode", range(5))
@pytest.mark.parametrize("native", [False, True])
def test_actual_hidden_hc_logical_rows_match_padded_three_rows_exactly(kept, mode, native):
    """Ten PSS groups and rank/MoE writeback retain identical valid-prefix bytes."""
    torch.manual_seed(188 + mode)
    rows, streams, dims = 3, 4, 2560
    h = torch.randn((rows, streams * dims), device="cuda", dtype=torch.float16) * 2
    inject = torch.randn((rows, streams), device="cuda", dtype=torch.float16) * 0.3
    branch = torch.randn((8, rows, dims), device="cuda") * 0.2
    y = torch.randn((rows, 11, dims), device="cuda") * 0.2
    wts = torch.randn((rows, 11), device="cuda").softmax(-1)
    scale = torch.randn(streams * dims, device="cuda") * 0.2 + 1
    up = torch.randn_like(h) * 0.2
    outputs = []
    for count in (kept, rows):
        out = h[:count].clone()
        pss = torch.empty((count, 10, streams), device="cuda")
        b = branch[:, :count].contiguous() if mode in (3, 4) else branch[0, :count].half().contiguous()
        invoke("hc_writeback", native, out, out, pss, streams, mode, branch=b,
               inject=inject[:count].contiguous(), y=y[:count].contiguous(), wts=wts[:count].contiguous())
        normed = torch.empty_like(out)
        xs = torch.empty((count, streams * dims // 32), device="cuda")
        invoke("hc_normed", native, out, pss, scale, normed, xs, streams, 1e-6)
        mixed = torch.empty((count, dims), device="cuda", dtype=torch.float16)
        sums = torch.empty((count, dims // 32), device="cuda")
        invoke("hc_mix", native, up[:count].contiguous(), normed, mixed, sums, streams)
        outputs.append((out, pss, normed, xs, mixed, sums))
    for actual, expected in zip(outputs[1], outputs[0]):
        integer = torch.int16 if actual.dtype == torch.float16 else torch.int32
        torch.testing.assert_close(actual[:kept].contiguous().view(integer), expected.contiguous().view(integer), atol=0, rtol=0)


@pytest.mark.parametrize("rows", [1, 2, 3, 18])
@pytest.mark.parametrize("native", [False, True])
def test_decode_ple_uses_serial_row_geometry_for_all_stored_tensors(rows, native):
    # This seed exposes a legacy batch-dependent FP16 convolution-history bin.
    torch.manual_seed(1002)
    streams, dims = 4, 2560
    h = torch.randn((rows, streams * dims), device="cuda", dtype=torch.float16) * 3
    keys = torch.randn_like(h) * 3
    vals = torch.randn((rows, dims), device="cuda", dtype=torch.float16)
    scale = torch.randn(streams * dims, device="cuda") * 0.2 + 1
    tail = torch.randn((9, streams * dims), device="cuda", dtype=torch.float16) * 0.4
    conv = torch.randn((streams * dims, 4), device="cuda", dtype=torch.float16) * 0.4
    expected, epss = torch.empty_like(h), torch.empty((rows, streams), device="cuda")
    for row in range(rows):
        section = slice(row, row + 1)
        invoke("ple_gate", False, keys[section], vals[section], h[section], scale, scale,
               expected[section], epss[section], 1e-6, streams)
    actual, apss = torch.empty_like(h), torch.empty_like(epss)
    live = torch.tensor([rows], device="cuda", dtype=torch.int32)
    invoke("ple_gate", native, keys, vals, h, scale, scale, actual, apss, 1e-6, streams,
           valid_rows=live, independent_rows=True)
    eout, enrow, aout, anrow = [torch.empty_like(h) for _ in range(4)]
    invoke("ple_conv", False, expected, epss, scale, tail, conv, h, eout, enrow, 1e-6, streams, 3)
    invoke("ple_conv", native, actual, apss, scale, tail, conv, h, aout, anrow, 1e-6, streams, 3)
    for a, e in ((actual, expected), (apss, epss), (aout, eout), (anrow, enrow)):
        integer = torch.int16 if a.dtype == torch.float16 else torch.int32
        torch.testing.assert_close(a.view(integer), e.view(integer), atol=0, rtol=0)


def test_captured_decode_ple_is_exact_for_advancing_live_row_counts():
    torch.manual_seed(1237)
    streams, dims, rows = 4, 2560, 3
    h = torch.randn((rows, streams * dims), device="cuda", dtype=torch.float16) * 3
    keys = torch.randn_like(h) * 3
    vals = torch.randn((rows, dims), device="cuda", dtype=torch.float16)
    scale = torch.randn(streams * dims, device="cuda") * 0.2 + 1
    actual, apss = torch.empty_like(h), torch.empty((rows, streams), device="cuda")
    live = torch.tensor([1], device="cuda", dtype=torch.int32)

    def run():
        invoke("ple_gate", True, keys, vals, h, scale, scale, actual, apss, 1e-6, streams,
               valid_rows=live, independent_rows=True)

    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for kept in (1, 3, 2, 1):
        live.fill_(kept)
        graph.replay()
        for row in range(kept):
            section = slice(row, row + 1)
            expected, epss = torch.empty_like(h[section]), torch.empty_like(apss[section])
            invoke("ple_gate", False, keys[section], vals[section], h[section], scale, scale,
                   expected, epss, 1e-6, streams)
            torch.testing.assert_close(actual[section].view(torch.int16), expected.view(torch.int16), atol=0, rtol=0)
            torch.testing.assert_close(apss[section].view(torch.int32), epss.view(torch.int32), atol=0, rtol=0)

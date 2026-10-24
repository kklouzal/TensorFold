"""SM70 family dispatch/state gates, using independent scalar oracles and real forward orchestration."""

import numpy as np
import pytest
import torch

from tensorfold.cuda.sm70 import family_ops as ops
from tensorfold.families.qwen4_exp.cuda import attention, gdn, glue

pytestmark = pytest.mark.skipif(torch.cuda.get_device_capability() != (7, 0), reason="SM70 FP16 family")


def test_half_hyperconnection_rank_reduction_and_roundings():
    torch.manual_seed(42)
    h = torch.randn((3, 1024), device="cuda", dtype=torch.float16)
    branches = torch.randn((8, 3, 256), device="cuda") * 0.1
    inject = torch.rand((3, 4), device="cuda", dtype=torch.float16)
    expected = h.cpu().numpy().copy().reshape(3, 4, 256)
    total = branches[0].cpu().numpy().copy()
    for rank in range(1, 8):
        total = total + branches[rank].cpu().numpy()
    branch = total.astype(np.float16)
    addition = (branch[:, None].astype(np.float32) * inject.cpu().numpy()[:, :, None]).astype(np.float16)
    expected = (expected.astype(np.float32) + addition.astype(np.float32)).astype(np.float16)
    pss = torch.empty((3, 1, 4), device="cuda")
    glue.hc_writeback(h, h, pss, 4, 3, branch=branches, inject=inject)
    np.testing.assert_array_equal(h.cpu().numpy(), expected.reshape(3, 1024))
    np.testing.assert_allclose(pss[:, 0].cpu().numpy(), (expected.astype(np.float64) ** 2).sum(-1),
                               rtol=3e-6, atol=2e-5)


def test_half_qsa_ties_causality_and_incomplete_block_tail():
    capacity, rows, heads, width = 20, 3, 3, 128
    scratch = attention.AttnScratch(rows, heads, width, capacity, "cuda", budget=8, ratio=4,
                                    dtype=torch.float16)
    iq = torch.zeros((rows, 4, width), device="cuda", dtype=torch.float16)
    pooled = torch.zeros((5, width), device="cuda", dtype=torch.float16)
    pos = torch.tensor([12], device="cuda", dtype=torch.int32)
    attention.qsa_rows(iq, pooled, pos, scratch, rows, context=15)
    # Equal scores retain the lowest two complete block IDs, followed by the current tail.
    np.testing.assert_array_equal(scratch.ids[:, :8].cpu().numpy(), np.tile(np.arange(8), (rows, 1)))
    q = torch.zeros((rows, heads, width), device="cuda", dtype=torch.float16)
    kc = torch.zeros((capacity, 1, width), device="cuda", dtype=torch.float16)
    vc = torch.arange(capacity, device="cuda", dtype=torch.float16)[:, None, None].expand(-1, 1, width).contiguous()
    output = attention.attention(q, kc, vc, pos, scratch, rows, width ** -0.5, context=15)
    for row in range(rows):
        selected = list(range(8)) + list(range(12, 13 + row))
        expected = np.float16(sum(selected) * float(np.float16(1.0 / len(selected))))
        np.testing.assert_array_equal(output[row].cpu().numpy(), np.full((heads, width), expected, np.float16))


def test_half_qsa_pool_block_crossing_prefill_chunks():
    torch.manual_seed(79)
    ikc = torch.randn((17, 128), device="cuda", dtype=torch.float16)
    pooled = torch.zeros((5, 128), device="cuda", dtype=torch.float16)
    scratch = attention.AttnScratch(7, 3, 128, 17, "cuda", budget=8, ratio=4, dtype=torch.float16)
    inv = torch.tensor([1e7 ** (-2 * i / 64) for i in range(32)], device="cuda")
    weight = torch.ones(128, device="cuda")
    attention.qsa_pool(ikc, pooled, torch.tensor([0], device="cuda"), weight, inv, 1e-6, scratch, 3)
    assert torch.count_nonzero(pooled) == 0
    attention.qsa_pool(ikc, pooled, torch.tensor([3], device="cuda"), weight, inv, 1e-6, scratch, 7)
    expected = []
    for block in range(2):
        raw = ikc[block * 4:(block + 1) * 4].cpu().numpy().astype(np.float32)
        mean = (((raw[0] + raw[1]) + raw[2]) + raw[3]) / 4
        mean = mean.astype(np.float16).astype(np.float64)
        normalized = (mean / np.sqrt(np.mean(mean ** 2) + 1e-6)).astype(np.float16)
        angle = block * 4 * inv.cpu().numpy().astype(np.float64)
        first, second = normalized[:32].astype(np.float64), normalized[32:64].astype(np.float64)
        rotated = normalized.copy()
        rotated[:32] = first * np.cos(angle) - second * np.sin(angle)
        rotated[32:64] = second * np.cos(angle) + first * np.sin(angle)
        expected.append(rotated)
    np.testing.assert_allclose(pooled[:2].cpu().numpy(), np.stack(expected), rtol=0.003, atol=0.003)
    assert torch.count_nonzero(pooled[2:]) == 0


@pytest.mark.parametrize("kept", [1, 2, 4])
def test_full_gdn_conv_norm_gates_and_replay_are_chunk_invariant(kept):
    torch.manual_seed(113)
    nk, nv, dk, dv, rows = 2, 6, 128, 128, 4
    channels, width = gdn.widths(nk, nv)
    p = torch.randn((rows, width), device="cuda", dtype=torch.float16) * 0.2
    conv = torch.randn((3, channels), device="cuda", dtype=torch.float16) * 0.2
    conv_w = torch.randn((channels, 4), device="cuda", dtype=torch.float16) * 0.2
    state = torch.randn((nv, dv, dk), device="cuda") * 0.01
    initial = state.clone()
    log = torch.randn(nv, device="cuda") * 0.1
    bias = torch.randn(nv, device="cuda") * 0.1
    norm = torch.ones(dv, device="cuda", dtype=torch.float16)
    scratch = gdn.GDNScratch(rows, "cuda", nk, nv, dtype=torch.float16)
    output = torch.empty((rows, nv * dv), device="cuda", dtype=torch.float16)
    sums = torch.empty((rows, nv * dv // 32), device="cuda")
    final = torch.empty_like(state)
    gdn.chain(p, conv, conv_w, state, log, bias, norm, 1e-6, rows, scratch, final, output, sums)
    reference_scratch = gdn.GDNScratch(rows, "cuda", nk, nv, dtype=torch.float16)
    reference_final, reference_output = torch.empty_like(final), torch.empty_like(output)
    ops.gdn_chain(p, conv, conv_w, state, log, bias, norm, 1e-6, rows, reference_scratch,
                  reference_final, reference_output, torch.empty_like(sums))
    torch.testing.assert_close(final, reference_final, rtol=4e-4, atol=1e-6)
    torch.testing.assert_close(output, reference_output, rtol=0.005, atol=0.003)
    replay = torch.empty_like(state)
    gdn.replay(state, scratch, kept, replay)
    serial, tail, outputs, prefix = state.clone(), conv.clone(), [], None
    for row in range(rows):
        one = gdn.GDNScratch(1, "cuda", nk, nv, dtype=torch.float16)
        out = torch.empty_like(output[:1])
        next_state = torch.empty_like(state)
        gdn.chain(p[row:row + 1], tail, conv_w, serial, log, bias, norm, 1e-6, 1, one, next_state,
                  out, torch.empty_like(sums[:1]))
        serial = next_state
        ops.shift_windows(tail[None], p[None, row:row + 1], 1, channels)
        outputs.append(out)
        if row + 1 == kept:
            prefix = serial.clone()
    torch.testing.assert_close(torch.cat(outputs), output, rtol=0, atol=0)
    torch.testing.assert_close(serial, final, rtol=0, atol=0)
    torch.testing.assert_close(prefix, replay, rtol=0, atol=0)
    torch.testing.assert_close(state, initial, rtol=0, atol=0)
    # Independent CPU FP64 recurrence over the saved normalized core inputs.
    reference = initial.cpu().double()
    for row in range(rows):
        for head in range(nv):
            key = scratch.k[row, head // 3].cpu().double()
            reference[head] *= float(scratch.g[row, head])
            residual = (scratch.v[row, head].cpu().double() - reference[head] @ key) * float(scratch.b[row, head])
            reference[head] += residual[:, None] * key[None]
    torch.testing.assert_close(final.cpu().double(), reference, rtol=3e-5, atol=3e-7)


def test_half_ple_conv_history_is_chunk_equivalent():
    torch.manual_seed(5)
    rows, streams, dims, taps, dilation = 5, 4, 256, 4, 3
    wide = streams * dims
    h = torch.randn((rows, wide), device="cuda", dtype=torch.float16)
    keys = torch.randn_like(h)
    values = torch.randn((rows, dims), device="cuda", dtype=torch.float16)
    gated = torch.empty_like(h)
    pss = torch.empty((rows, streams), device="cuda")
    weight = torch.ones(wide, device="cuda")
    glue.ple_gate(keys, values, h, weight, weight, gated, pss, 1e-6, streams)
    old = torch.randn(((taps - 1) * dilation, wide), device="cuda", dtype=torch.float16)
    conv_w = torch.randn((wide, taps), device="cuda", dtype=torch.float16) * 0.1
    result, nrow = torch.empty_like(h), torch.empty_like(h)
    glue.ple_conv(gated, pss, weight, old, conv_w, h, result, nrow, 1e-6, streams, dilation)
    serial, tail, normalized = [], old.clone(), []
    for row in range(rows):
        output, current = torch.empty_like(h[:1]), torch.empty_like(h[:1])
        glue.ple_conv(gated[row:row + 1], pss[row:row + 1], weight, tail, conv_w, h[row:row + 1],
                      output, current, 1e-6, streams, dilation)
        ops.shift_windows(tail[None], current[None], 1, wide)
        serial.append(output)
        normalized.append(current)
    torch.testing.assert_close(torch.cat(serial), result, rtol=0, atol=0)
    torch.testing.assert_close(torch.cat(normalized), nrow, rtol=0, atol=0)


def _small_weights(include_ple=False, *, hidden=256, lowrank=32, head_dim=128):
    from tensorfold.families.qwen4_exp.cuda.weight_types import AttnW, Config, GDNW, HC, LayerW, MoEW, PLEW, Weights

    torch.manual_seed(183)
    dims, streams, low, vocab, nk, nv, heads, hk, hd = hidden, 4, lowrank, 512, 2, 6, 3, 1, head_dim
    config = Config(dims, 2, ["linear", "attention"], vocab, 1e-6, heads, hk, hd, 1e7, 64,
                    nk, nv, 128, 128, 4, 3, 2, 80, 80, streams, low, 4, 128, 8, 4,
                    [], 512, 4, 3, 8, 64, 16, 1, 42, 0, (0,), 32, 4,
                    quant="gguf", execution_dtype="float16")
    if include_ple:
        config.ple_layers = [0]

    class Linear:
        kernel = "gguf"

        def __init__(self, n, k):
            self.n, self.k = n, k
            self.weight = torch.randn((n, k), device="cuda", dtype=torch.float16) * 0.01

        def matmul(self, x, out=None, f32=False):
            # One-row FP32 oracle keeps each row's arithmetic independent of batch shape.
            result = torch.cat([x[row:row + 1].float() @ self.weight.float().T for row in range(x.shape[0])])
            if out is None:
                return result if f32 else result.half()
            out.copy_(result)
            return out

        def __call__(self, x, out=None):
            return self.matmul(x, out, f32=out is not None and out.dtype == torch.float32)

        prefill = __call__

        def partials(self, x):
            return self.matmul(x, f32=True)[None]

    class Embedding:
        def __init__(self):
            self.values = torch.randn((vocab, dims), device="cuda", dtype=torch.float16) * 0.3

        def lookup(self, ids, copies=1, out=None):
            values = self.values[ids.long()].repeat(1, copies)
            if out is None:
                return values
            out.copy_(values)
            return out

    class Experts:
        kernel = "gguf"

        def forward(self, x, picks, out=None):
            value = x[:, None].float() * (picks[:, :, None].float() + 1) * 0.01
            if out is None:
                return value
            out.copy_(value)
            return out

    class Table:
        bits = 16
        weight_scale = 1.0

        def gather(self, ids):
            indices = np.asarray(ids, dtype=np.int64).reshape(-1, 1)
            columns = np.arange(32, dtype=np.int64)[None]
            return (((indices + columns) % 17 - 8) * 0.02).astype(np.float16)

    def hc(inject):
        return HC(Linear(low + (streams if inject else 0), streams * dims), Linear(streams * dims, low),
                  torch.ones(streams * dims, device="cuda"), inject)

    conv, width = gdn.widths(nk, nv)
    recurrent = GDNW(Linear(width, dims), torch.randn((conv, 4), device="cuda", dtype=torch.float16) * 0.1,
                    torch.zeros(nv, device="cuda"), torch.zeros(nv, device="cuda"),
                    torch.ones(128, device="cuda", dtype=torch.float16), Linear(dims, nv * 128))
    attention_width = heads * 2 * hd + 2 * hk * hd + 5 * 128
    att = AttnW(Linear(attention_width, dims), torch.ones(hd, device="cuda"), torch.ones(hd, device="cuda"),
                torch.ones(128, device="cuda"), torch.ones(128, device="cuda"), Linear(dims, heads * hd))
    layers = []
    for index, linear in enumerate((True, False)):
        moe = MoEW(torch.randn((4, dims), device="cuda", dtype=torch.float16) * 0.03, Experts())
        ple = None
        if include_ple and index == 0:
            ple = PLEW(Table(), Linear(streams * dims, config.ple_dim), Linear(dims, config.ple_dim),
                       torch.ones(streams * dims, device="cuda"), torch.ones(streams * dims, device="cuda"),
                       torch.ones(streams * dims, device="cuda"),
                       torch.randn((streams * dims, 4), device="cuda", dtype=torch.float16) * 0.1,
                       config.ngram(0))
        layers.append(LayerW(index, linear, hc(True), hc(True), recurrent if linear else None,
                             None if linear else att, moe, ple))
    inverse = torch.tensor([1e7 ** (-2 * i / 64) for i in range(32)], device="cuda")
    return Weights(config, (Embedding(),), layers, hc(False), Linear(vocab, dims), inverse,
                   meta={"execution_dtype": "float16", "execution_provider": "sm70", "world": 1})


@pytest.mark.parametrize("include_ple", [False, True])
def test_existing_family_forward_prefill_decode_and_snapshot_half_identity(include_ple):
    from tensorfold.families.qwen4_exp.cuda.forward import commit, forward
    from tensorfold.families.qwen4_exp.cuda.state import Buffers, State

    w = _small_weights(include_ple)
    prompt = [4, 17, 3, 99, 33]
    serial = State(w, 16, 8, "fp16")
    batch = State(w, 16, 8, "fp16")
    pref = State(w, 16, 2, "fp16")
    buffers = Buffers(w, 8, 16)
    pbuf = Buffers(w, 8, 16, prefill=True)
    assert buffers.h.dtype == batch.conv.dtype == batch.kc[0].k.dtype == torch.float16
    assert batch.rec.dtype == torch.float32
    whole = forward(w, batch, buffers, prompt).clone()
    commit(w, batch, buffers, len(prompt), len(prompt))
    outputs = []
    for token in prompt:
        outputs.append(forward(w, serial, buffers, [token]).clone())
        commit(w, serial, buffers, 1, 1)
    prompt_result = forward(w, pref, pbuf, prompt).clone()
    commit(w, pref, pbuf, len(prompt), len(prompt))
    torch.testing.assert_close(torch.cat(outputs), whole, rtol=0, atol=0.001)
    torch.testing.assert_close(prompt_result[0], whole[-1], rtol=0, atol=0.001)
    torch.testing.assert_close(batch.rec[batch.cur[0]], serial.rec[serial.cur[0]], rtol=0, atol=0)
    torch.testing.assert_close(pref.rec[pref.cur[0]], serial.rec[serial.cur[0]], rtol=0, atol=0)
    torch.testing.assert_close(batch.kc[0].k[:5], serial.kc[0].k[:5], rtol=0, atol=0)
    restored = batch.clone()
    restored.restore(serial.snapshot())
    assert restored.kv_dtype == "fp16" and restored.kv_working_policy == "stored-basis-fp16-fp32acc-v1"
    assert restored.pos == serial.pos
    with pytest.raises(ValueError, match="explicit fp16"):
        State(w, 16, 8, "bf16")


def test_half_prefill_cut_snapshot_can_resume_with_ple_and_gdn():
    from tensorfold.families.qwen4_exp.cuda.forward import Cut, commit, cut_snapshot, forward
    from tensorfold.families.qwen4_exp.cuda.state import Buffers, State

    w = _small_weights(True)
    tokens = [4, 17, 3, 99, 33]
    source, resumed = State(w, 16, 8, "fp16"), State(w, 16, 8, "fp16")
    pref = Buffers(w, 8, 16, prefill=True)
    cut = Cut(2)
    target = forward(w, source, pref, tokens, cut=cut).clone()
    snapshot = cut_snapshot(w, source, pref, cut, 0)
    commit(w, source, pref, len(tokens), len(tokens))
    resumed.copy_prefix(source, 2, 0)
    resumed.restore(snapshot)
    actual = forward(w, resumed, pref, tokens[2:]).clone()
    commit(w, resumed, pref, 3, 3)
    torch.testing.assert_close(actual, target, rtol=0, atol=0.001)
    torch.testing.assert_close(resumed.rec[resumed.cur[0]], source.rec[source.cur[0]], rtol=0, atol=0)
    torch.testing.assert_close(resumed.conv, source.conv, rtol=0, atol=0)
    torch.testing.assert_close(resumed.ple_tail, source.ple_tail, rtol=0, atol=0)
    np.testing.assert_array_equal(resumed.ple_history, source.ple_history)


def test_native_gdn_large_prompt_and_multiple_kept_prefixes_match_unsplit_chain():
    from tensorfold.families.qwen4_exp.cuda.forward import Cut, _prefill_chain
    from tensorfold.families.qwen4_exp.cuda.state import Buffers, State

    w = _small_weights()
    c, rows = w.cfg, 2049
    state = State(w, rows + 1, 2, "fp16")
    buf = Buffers(w, rows, rows + 1, prefill=True)
    torch.manual_seed(919)
    state.rec[0].normal_(std=0.01)
    state.conv.normal_(std=0.1)
    buf.proj.normal_(std=0.1)
    initial, convolution = state.rec[0, 0].clone(), state.conv[0].clone()
    recurrent = w.layers[0].gdn
    scratch = gdn.GDNScratch(rows, w.device, c.nk, c.nv, dtype=torch.float16)
    expected, expected_state = torch.empty_like(buf.gout), torch.empty_like(initial)
    gdn.chain(buf.proj[0], convolution, recurrent.conv, initial, recurrent.a_log, recurrent.dt_bias,
              recurrent.norm, c.eps, rows, scratch, expected_state, expected, torch.empty_like(buf.gxs))
    cuts = [Cut(point) for point in (511, 1025, 2048)]
    _prefill_chain(recurrent, state, 0, buf, 0, rows, c, cuts)
    torch.testing.assert_close(buf.gout, expected, atol=0, rtol=0)
    torch.testing.assert_close(state.rec[state.cur[0], 0], expected_state, atol=0, rtol=0)
    torch.testing.assert_close(state.conv[0], buf.proj[0, -3:, :c.conv_dim], atol=0, rtol=0)
    prefix = torch.empty_like(initial)
    for cut in cuts:
        gdn.replay(initial, scratch, cut.row, prefix)
        torch.testing.assert_close(cut.rec[0], prefix, atol=0, rtol=0)
        torch.testing.assert_close(cut.conv[0], buf.proj[0, cut.row - 3:cut.row, :c.conv_dim], atol=0, rtol=0)


def test_half_existing_compute_keeps_multiple_request_states_independent():
    from tensorfold.families.qwen4_exp.cuda.forward import commit, compute, forward, stage
    from tensorfold.families.qwen4_exp.cuda.state import Buffers, State

    w = _small_weights(True)
    left, right = State(w, 16, 8, "fp16"), State(w, 16, 8, "fp16")
    buf = Buffers(w, 8, 16)
    for state, prefix in ((left, [13]), (right, [8, 19, 4])):
        forward(w, state, buf, prefix)
        commit(w, state, buf, len(prefix), len(prefix))
    left_ref, right_ref = left.clone(), right.clone()
    windows = [(left, [71, 33]), (right, [18, 91, 2])]
    segs = stage(w, buf, windows)
    combined = compute(w, segs, buf).clone()
    for state, begin, end in segs:
        commit(w, state, buf, end - begin, end - begin, at=begin)
    for state, reference, tokens, begin in ((left, left_ref, [71, 33], 0), (right, right_ref, [18, 91, 2], 2)):
        expected = forward(w, reference, buf, tokens).clone()
        commit(w, reference, buf, len(tokens), len(tokens))
        torch.testing.assert_close(combined[begin:begin + len(tokens)], expected, rtol=0, atol=0.001)
        torch.testing.assert_close(state.rec[state.cur[0]], reference.rec[reference.cur[0]], rtol=0, atol=0)
        torch.testing.assert_close(state.kc[0].k[:state.pos], reference.kc[0].k[:reference.pos], rtol=0, atol=0)
        torch.testing.assert_close(state.ple_tail, reference.ple_tail, rtol=0, atol=0)


def test_native_half_gdn_empty_rows_and_prelaunch_disjointness():
    nk, nv = 2, 6
    channels, width = gdn.widths(nk, nv)
    p = torch.zeros((0, width), device="cuda", dtype=torch.float16)
    conv = torch.zeros((3, channels), device="cuda", dtype=torch.float16)
    weights = torch.zeros((channels, 4), device="cuda", dtype=torch.float16)
    initial = torch.randn((nv, 128, 128), device="cuda")
    final = torch.empty_like(initial)
    output = torch.empty((0, nv * 128), device="cuda", dtype=torch.float16)
    sums = torch.empty((0, nv * 4), device="cuda")
    log, bias = torch.zeros(nv, device="cuda"), torch.zeros(nv, device="cuda")
    norm = torch.ones(128, device="cuda", dtype=torch.float16)
    scratch = gdn.GDNScratch(0, "cuda", nk, nv, dtype=torch.float16)
    gdn.chain(p, conv, weights, initial, log, bias, norm, 1e-6, 0, scratch, final, output, sums)
    torch.testing.assert_close(final, initial, rtol=0, atol=0)
    with pytest.raises(RuntimeError, match="disjoint"):
        gdn.chain(p, conv, weights, initial, log, bias, norm, 1e-6, 0, scratch, initial, output, sums)
    with pytest.raises(RuntimeError, match="epsilon"):
        gdn.chain(p, conv, weights, initial, log, bias, norm, float("nan"), 0, scratch, final, output, sums)


def test_int8_bytes_match_existing_codec_and_half_decode_scalar_oracle():
    from tensorfold.families.qwen4_exp.cuda.kvcache import quantize_ref

    torch.manual_seed(419)
    value = torch.randn((7, 1, 128), device="cuda", dtype=torch.float16)
    encoded, scales = ops.quantize_int8(value)
    original, original_scales, _, _ = quantize_ref(value, value, 8)
    torch.testing.assert_close(encoded, original, rtol=0, atol=0)
    torch.testing.assert_close(scales, original_scales, rtol=0, atol=0)
    expected = (encoded.cpu().numpy().astype(np.float32).reshape(7, 1, 4, 32) + 0.5)
    expected *= scales.cpu().numpy().astype(np.float32)[..., None] / 128
    np.testing.assert_array_equal(ops.dequantize_int8(encoded, scales).cpu().numpy(), expected.reshape(7, 1, 128).astype(np.float16))


@pytest.mark.parametrize("key,value", [("int8", "int8"), ("fp16", "int8"), ("int8", "fp16")])
def test_half_integer_kv_forward_prefix_copy_and_state_are_chunk_equivalent(key, value):
    from tensorfold.families.qwen4_exp.cuda.forward import commit, forward
    from tensorfold.families.qwen4_exp.cuda.state import Buffers, State
    from tensorfold.families.qwen4_exp.kv_formats import get_pair

    w = _small_weights(True)
    pair = get_pair("fp16", key, value)
    serial, batch = [State(w, 24, 12, kv_pair=pair) for _ in range(2)]
    buf = Buffers(w, 12, 24)
    tokens = [4, 17, 3, 99, 33, 7, 31, 81, 2, 5, 9, 6]
    whole = forward(w, batch, buf, tokens).clone()
    commit(w, batch, buf, len(tokens), len(tokens))
    outputs = []
    for token in tokens:
        outputs.append(forward(w, serial, buf, [token]).clone())
        commit(w, serial, buf, 1, 1)
    torch.testing.assert_close(torch.cat(outputs), whole, rtol=0, atol=0.002)
    for name in ("k", "v", "ks", "vs"):
        torch.testing.assert_close(getattr(batch.kc[0], name), getattr(serial.kc[0], name), rtol=0, atol=0)
    copied = State(w, 24, 12, kv_pair=pair)
    copied.copy_prefix(batch, batch.pos, 0)
    copied.restore(batch.snapshot())
    continued = forward(w, copied, buf, [11]).clone()
    expected = forward(w, batch, buf, [11]).clone()
    torch.testing.assert_close(continued, expected, rtol=0, atol=0)


@pytest.mark.parametrize("cache", ["fp16", "int8"])
def test_whole_half_forward_graph_replays_advancing_positions_and_both_gdn_parities(cache):
    from types import SimpleNamespace
    from tensorfold.families.qwen4_exp.cuda.forward import commit, forward
    from tensorfold.families.qwen4_exp.cuda.graphs import Graphs
    from tensorfold.families.qwen4_exp.cuda.state import Buffers, State

    w = _small_weights(True)
    graph_state, eager_state = [State(w, 24, 12, cache) for _ in range(2)]
    graph_buf, eager_buf = Buffers(w, 12, 24), Buffers(w, 12, 24)
    prefix = [4, 17, 9, 31, 16, 25, 13, 15, 7]
    for state, buf in ((graph_state, graph_buf), (eager_state, eager_buf)):
        forward(w, state, buf, prefix)
        commit(w, state, buf, len(prefix), len(prefix))
    graph = Graphs(SimpleNamespace(w=w, st=graph_state, buf=graph_buf, mbuf=None), max_rows=2)
    for tokens in ([71, 33], [18, 91], [2, 8]):
        expected = forward(w, eager_state, eager_buf, tokens).clone()
        actual = graph.forward(tokens).clone()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        commit(w, graph_state, graph_buf, len(tokens), len(tokens))
        commit(w, eager_state, eager_buf, len(tokens), len(tokens))
        torch.testing.assert_close(graph_state.rec[graph_state.cur[0]], eager_state.rec[eager_state.cur[0]], rtol=0, atol=0)
        torch.testing.assert_close(graph_state.ple_tail, eager_state.ple_tail, rtol=0, atol=0)
    assert graph.captures == 2

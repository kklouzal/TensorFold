"""Qualified SM70 router preserves every original serial FP32 logit and Half gate boundary."""
from types import SimpleNamespace as NS

import pytest
import torch

from tensorfold.cuda import moe
from tensorfold.cuda.sm70 import routing_native

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0),
                                reason="SM70 exact serial router")


def serial_logits(x, weight):
    value, coefficients = x.float(), weight.float()
    out = torch.empty((x.shape[0], weight.shape[0]), dtype=torch.float32, device=x.device)
    for row in range(x.shape[0]):
        out[row:row+1].copy_(value[row:row+1] @ coefficients.T)
    return out


def native(x, weight, out=None):
    if out is None:
        out = torch.empty((x.shape[0], weight.shape[0]), dtype=torch.float32, device=x.device)
    return routing_native.logits(x, weight, out)


def exact(a, b):
    torch.testing.assert_close(a.contiguous().view(torch.int32), b.contiguous().view(torch.int32), rtol=0, atol=0)


def routes(logits):
    picks = torch.argsort(logits[:, :512], dim=1, stable=True, descending=True)[:, :10]
    probabilities = torch.softmax(logits[:, :512].gather(1, picks), dim=1).half().float()
    shared = torch.sigmoid(logits[:, 512:].half().float()).half().float()
    return picks, probabilities, shared


@pytest.mark.parametrize("rows", [1, 2, 3, 4, 6, 8, 18, 32])
@pytest.mark.parametrize("seed", [0, 113, 1237])
def test_every_logit_and_selected_half_weight_matches_original_serial(rows, seed):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn((rows, 2560), generator=generator, device="cuda", dtype=torch.float16) * .7
    weight = (torch.randn((513, 2560), generator=generator, device="cuda", dtype=torch.float16) * .03).float()
    expected = serial_logits(x, weight)
    result = native(x, weight)
    exact(result, expected)
    for actual, original in zip(routes(result), routes(expected)):
        torch.testing.assert_close(actual, original, rtol=0, atol=0)


@pytest.mark.parametrize("rows", [2, 3, 6, 8, 18])
def test_seed113_adversarial_half_bin_proves_old_gemm_is_not_authority(rows):
    generator = torch.Generator(device="cuda").manual_seed(113)
    three = torch.randn((3, 2560), generator=generator, device="cuda", dtype=torch.float16) * .7
    weight = (torch.randn((513, 2560), generator=generator, device="cuda", dtype=torch.float16) * .03).float()
    x = three.repeat((rows + 2) // 3, 1)[:rows]
    serial = serial_logits(x, weight)
    # Keep the independently established R2/R3 failure geometry as the
    # negative control; a larger GEMM may choose another vendor algorithm.
    old = three[:min(rows, 3)].float() @ weight.T
    _, original_weights, _ = routes(serial)
    _, old_weights, _ = routes(old)
    assert original_weights[1, 0].item() == .1453857421875
    assert old_weights[1, 0].item() != original_weights[1, 0].item()
    result = native(x, weight)
    exact(result, serial)
    for actual, original in zip(routes(result), routes(serial)):
        torch.testing.assert_close(actual, original, rtol=0, atol=0)


@pytest.mark.parametrize("rows", [0, 3, 18])
def test_channel_contiguous_padded_rows_match_serial_and_preserve_neighbors(rows):
    generator = torch.Generator(device="cuda").manual_seed(991)
    storage = torch.randn((rows, 2567), dtype=torch.float16, device="cuda", generator=generator)
    x = storage[:, :2560]
    weight = (torch.randn((513, 2560), dtype=torch.float16, device="cuda", generator=generator) * .03).float()
    backing = torch.full((rows * 513 + 256,), -17, dtype=torch.float32, device="cuda")
    out = backing[:rows * 513].view(rows, 513)
    expected = serial_logits(x, weight)
    assert native(x, weight, out) is out
    exact(out, expected)
    assert torch.all(backing[rows * 513:] == -17)


@pytest.mark.parametrize("rows", [1, 6, 8, 18])
def test_graph_replays_live_inputs_and_weight_contents_using_the_original_r1_oracle(rows):
    torch.manual_seed(317)
    x = torch.randn((rows, 2560), device="cuda", dtype=torch.float16)
    weight = (torch.randn((513, 2560), device="cuda", dtype=torch.float16) * .02).float()
    output = torch.empty((rows, 513), device="cuda")
    call = native
    call(x, weight, output)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call(x, weight, output)
    for scale in (.75, -.125):
        x.mul_(scale)
        graph.replay()
        torch.cuda.synchronize()
        exact(output, serial_logits(x, weight))


@pytest.mark.parametrize("kind", ["zero", "negative_zero", "infinity", "nan"])
def test_special_arithmetic_is_compared_as_raw_float32_bits(kind):
    x = torch.ones((3, 2560), dtype=torch.float16, device="cuda")
    weight = torch.ones((513, 2560), dtype=torch.float32, device="cuda")
    if kind == "zero":
        x.zero_()
    elif kind == "negative_zero":
        x.fill_(-0.0)
    elif kind == "infinity":
        x[1, 17] = torch.inf
        weight[3, 24] = -torch.inf
    else:
        x[1, 17] = torch.nan
        weight[3, 24] = torch.nan
    exact(native(x, weight), serial_logits(x, weight))


def test_decode_dispatch_selects_matching_static_float_router_and_retains_r1_and_prompt(monkeypatch):
    generator = torch.Generator(device="cuda").manual_seed(113)
    x = torch.randn((3, 2560), generator=generator, dtype=torch.float16, device="cuda") * .7
    weight = (torch.randn((513, 2560), generator=generator, dtype=torch.float16, device="cuda") * .03).float()
    original = routing_native.logits
    calls = []

    def selected(source, coefficients, output):
        calls.append(source.shape[0])
        return original(source, coefficients, output)

    monkeypatch.setattr(routing_native, "_NATIVE_LOGITS", True)
    monkeypatch.setattr(routing_native, "logits", selected)
    exact(moe.router(x, weight, independent_rows=True), serial_logits(x, weight))
    assert calls == [3]
    exact(moe.router(x[:1], weight, independent_rows=True), serial_logits(x[:1], weight))
    exact(moe.router(x, weight), x.float() @ weight.T)
    exact(moe.router(x, weight.half(), independent_rows=True), serial_logits(x, weight))
    assert calls == [3]


def test_native_binding_rejects_alias_or_unqualified_storage_before_launch():
    x = torch.zeros((3, 2560), dtype=torch.float16, device="cuda")
    weight = torch.zeros((513, 2560), dtype=torch.float32, device="cuda")
    with pytest.raises(RuntimeError, match="overlap"):
        routing_native.logits(x, weight, weight.view(-1)[:3 * 513].view(3, 513))
    with pytest.raises(RuntimeError, match="weights"):
        routing_native.logits(x, weight.half(), torch.empty((3, 513), device="cuda"))
    with pytest.raises(RuntimeError, match="output"):
        routing_native.logits(x, weight, torch.empty((3, 513), dtype=torch.float16, device="cuda"))


def actual_router_weights():
    from test_flashnext_sm70_family import _small_weights
    from test_sm70_gguf_prepared import dense_fixture

    w = dense_fixture(_small_weights(include_ple=True, hidden=2560, lowrank=320, head_dim=256))
    w.cfg.experts, w.cfg.top_k = 512, 10
    for layer in w.layers:
        layer.moe.router = (torch.randn((513, 2560), dtype=torch.float16, device="cuda") * .03).float()
    return w


@pytest.mark.parametrize("cache", ["fp16", "int8"])
@pytest.mark.parametrize("prepared", [False, True])
def test_exact_router_preserves_actual_geometry_full_logits_partial_commits_and_prefix(cache, prepared, monkeypatch):
    import numpy as np
    from tensorfold.cuda.sm70 import family_ops
    from tensorfold.families.qwen4_exp.cuda import padded_linears
    from tensorfold.families.qwen4_exp.cuda.forward import commit, forward
    from tensorfold.families.qwen4_exp.cuda.state import Buffers, State

    monkeypatch.setattr(family_ops, "_NATIVE_FAMILY", True)
    monkeypatch.setattr(family_ops, "_NATIVE_ATTENTION", True)
    monkeypatch.setattr(padded_linears, "_ENABLED", prepared)
    w = actual_router_weights()
    states = [State(w, 64, 18, cache) for _ in range(2)]
    buffers = [Buffers(w, 18, 64, moe_prefill=True) for _ in range(2)]
    for tokens, keep in ((list(range(1, 7)), 3), ([11, 12, 13], 2), (list(range(21, 39)), 11), ([41], 1)):
        outputs = []
        for enabled, state, buf in zip((False, True), states, buffers):
            monkeypatch.setattr(routing_native, "_NATIVE_LOGITS", enabled)
            outputs.append(forward(w, state, buf, tokens).clone())
            commit(w, state, buf, len(tokens), keep)
        exact(*outputs)
        exact(buffers[0].streams[:len(tokens)], buffers[1].streams[:len(tokens)])
        a, b = states
        assert a.pos == b.pos and a.cur == b.cur
        for index, parity in enumerate(a.cur):
            exact(a.rec[parity, index], b.rec[b.cur[index], index])
        for name in ("conv", "ple_tail"):
            torch.testing.assert_close(getattr(a, name), getattr(b, name), rtol=0, atol=0)
        for ka, kb in zip(a.kc, b.kc):
            for name in ("k", "v", "ks", "vs"):
                torch.testing.assert_close(getattr(ka, name)[:a.pos], getattr(kb, name)[:b.pos], rtol=0, atol=0)
        for name in ("ikc", "pooled"):
            end = a.pos if name == "ikc" else a.pos // a.ratio
            for va, vb in zip(getattr(a, name), getattr(b, name)):
                torch.testing.assert_close(va[:end], vb[:end], rtol=0, atol=0)
        np.testing.assert_array_equal(a.ple_history, b.ple_history)
    copies = [State(w, 64, 18, cache) for _ in range(2)]
    for source, target in zip(states, copies):
        target.copy_prefix(source, source.pos, 0)
        target.restore(source.snapshot())
    torch.testing.assert_close(copies[0].rec, copies[1].rec, rtol=0, atol=0)
    torch.testing.assert_close(copies[0].conv, copies[1].conv, rtol=0, atol=0)


def test_exact_router_bound_six_lane_capture_matches_logical_rows_at_actual_geometry(monkeypatch):
    from test_flashnext_sm70_bound_graphs import _qualify_six_lanes
    from tensorfold.cuda.sm70 import family_ops
    from tensorfold.families.qwen4_exp.cuda import padded_linears

    monkeypatch.setattr(family_ops, "_NATIVE_FAMILY", True)
    monkeypatch.setattr(family_ops, "_NATIVE_ATTENTION", True)
    monkeypatch.setattr(padded_linears, "_ENABLED", True)
    monkeypatch.setattr(routing_native, "_NATIVE_LOGITS", True)
    _qualify_six_lanes(actual_router_weights(), "int8")


def prepared_fixture(x, weight, role):
    """Real owner/role binding; reuse the unchanged serial GEMV oracle above."""
    count = x.shape[0]
    cap = routing_native.prepare_serial_router(x.device, {'decode': count, 'mtp': count})
    other = weight.clone()
    target, draft = (weight, other) if role == 'decode' else (other, weight)
    weights = NS(device=x.device, cfg=NS(hidden=2560, experts=512, top_k=10, execution_dtype='float16'),
        layers=[NS(moe=NS(router=target))], mtp=NS(layer=NS(moe=NS(router=draft))))
    cap.bind_weights(weights)
    buffers = NS(owner_role=role, rows=count, prefill=False, mixed=x,
                 moe=NS(logits=torch.empty((count, 513), dtype=torch.float32, device=x.device)))
    bound = cap.bind_buffers(buffers)
    return cap, bound, buffers.moe.logits


def expert_weight_packet(logits, *, reference):
    buf = NS(act=torch.empty((), dtype=torch.float16, device=logits.device), selection=None,
        pick=torch.empty((logits.shape[0], 11), dtype=torch.int32, device=logits.device),
        wts=torch.empty((logits.shape[0], 11), dtype=torch.float32, device=logits.device))
    (moe._torch_select_rows if reference else moe.select_rows)(logits, buf, 10, 512)
    return buf.pick, buf.wts


def check_prepared(x, weight, role):
    cap, bound, out = prepared_fixture(x, weight, role)
    before = out._version
    expected = serial_logits(x, weight)
    invoked = []
    native_call = cap.logits
    def traced(*args):
        invoked.append((role, args[0].shape[0]))
        return native_call(*args)
    cap.logits = traced
    if not bound.run(x, weight, out):
        moe.router(x, weight, out, independent_rows=True)
    torch.cuda.synchronize()
    exact(out, expected)
    assert out._version-before == x.shape[0]
    assert invoked == ([(role, x.shape[0])] if x.shape[0] > 1 else [])
    actual, original = expert_weight_packet(out, reference=False), expert_weight_packet(expected, reference=True)
    assert torch.equal(actual[0], original[0])
    exact(actual[1], original[1])
    torch.cuda.synchronize()
    cap.close()
    return out


@pytest.mark.parametrize('rows', [1, 2, 3, 8, 24, 32, 33, 37, 64, 120, 256])
@pytest.mark.parametrize('role', ['decode', 'mtp'])
@pytest.mark.parametrize('weight_precision', ['half-derived', 'float32'])
@torch.no_grad()
def test_prepared_router_exact_serial_logits_and_expert_weight_packet(rows, role, weight_precision):
    generator = torch.Generator(device='cuda').manual_seed(1407)
    x = torch.randn((rows, 2560), generator=generator, device='cuda', dtype=torch.float16)*.7
    dtype = torch.float16 if weight_precision == 'half-derived' else torch.float32
    weight = (torch.randn((513, 2560), generator=generator, device='cuda', dtype=dtype)*.03).float()
    check_prepared(x, weight, role)


@pytest.mark.parametrize('rows', [2, 33, 37, 120, 256])
@torch.no_grad()
def test_prepared_router_seed113_half_boundary_packet_keeps_negative_gemm_control(rows):
    generator = torch.Generator(device='cuda').manual_seed(113)
    three = torch.randn((3, 2560), generator=generator, device='cuda', dtype=torch.float16)*.7
    weight = (torch.randn((513, 2560), generator=generator, device='cuda', dtype=torch.float16)*.03).float()
    x = three.repeat((rows+2)//3, 1)[:rows]
    expected = serial_logits(x, weight)
    assert routes(expected)[1][1, 0].item() == .1453857421875
    assert routes(three.float() @ weight.T)[1][1, 0].item() != routes(expected)[1][1, 0].item()
    check_prepared(x, weight, 'mtp')


@pytest.mark.parametrize('kind', ['ties', 'negative-zero', 'infinity', 'nan'])
@torch.no_grad()
def test_prepared_router_adverse_bits_and_stable_expert_weights(kind):
    x = torch.ones((37, 2560), dtype=torch.float16, device='cuda')
    weight = torch.ones((513, 2560), dtype=torch.float32, device='cuda')
    if kind == 'ties':
        weight.zero_()
    elif kind == 'negative-zero':
        x.fill_(-0.)
    elif kind == 'infinity':
        x[33, 17] = torch.inf
        weight[3, 24] = -torch.inf
    else:
        x[33, 17] = torch.nan
        weight[3, 24] = torch.nan
    check_prepared(x, weight, 'decode')


@pytest.mark.parametrize('rows', [5, 33, 120, 256])
@torch.no_grad()
def test_prepared_router_graph_uses_live_bound_input_and_weight_owners(rows):
    generator = torch.Generator(device='cuda').manual_seed(307)
    x = torch.randn((rows, 2560), generator=generator, device='cuda', dtype=torch.float16)
    weight = torch.randn((513, 2560), generator=generator, device='cuda', dtype=torch.float32)*.02
    _cap, bound, out = prepared_fixture(x, weight, 'mtp')
    assert bound.run(x, weight, out)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        assert bound.run(x, weight, out)
    for scale in (.75, -.125):
        x.mul_(scale)
        weight.mul_(.5)
        graph.replay()
        torch.cuda.synchronize()
        exact(out, serial_logits(x, weight))
    del graph
    _cap.close()


@torch.no_grad()
def test_prepared_native_extent_is_separate_and_whole_output_aliases_reject():
    x = torch.zeros((33, 2560), dtype=torch.float16, device='cuda')
    weight = torch.zeros((513, 2560), dtype=torch.float32, device='cuda')
    out = torch.empty((33, 513), device='cuda')
    cap = routing_native.prepare_serial_router(x.device, {'decode': 33})
    with pytest.raises(RuntimeError, match='at most32'):
        cap.module.logits(x, weight, out)
    cap.module.prepared_logits(x, weight, out)
    for rows in (0, 1, 257):
        with pytest.raises(RuntimeError, match='2..256'):
            cap.module.prepared_logits(torch.empty((rows, 2560), dtype=torch.float16, device=x.device),
                weight, torch.empty((rows, 513), device=x.device))
    with pytest.raises(RuntimeError, match='overlap'):
        cap.module.prepared_logits(x, weight, weight.reshape(-1)[:33*513].reshape(33, 513))
    torch.cuda.synchronize()
    cap.close()

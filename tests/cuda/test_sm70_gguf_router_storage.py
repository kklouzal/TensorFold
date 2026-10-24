"""Router cast elimination preserves logits, committed state and weight budgets."""

import weakref

import numpy as np
import pytest
import torch

from gguf_fixture import tiny_model
from test_flashnext_sm70_family import _small_weights
from tensorfold.cuda import moe
from tensorfold.cuda.sm70 import family_ops
from tensorfold.families.qwen4_exp.cuda import gguf_weights

pytestmark = pytest.mark.skipif(torch.cuda.get_device_capability() != (7, 0), reason="SM70 router storage")


def exact(actual, expected):
    torch.testing.assert_close(actual.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8),
                               rtol=0, atol=0)


def test_tiny_router_tensor_is_float32_budgeted_and_does_not_retain_cached_parts(tmp_path, monkeypatch):
    path = tiny_model(tmp_path / "tiny.gguf")
    parts = []
    original = gguf_weights._array

    def read(plan, reader, device):
        if plan.source.name == "output_hc_norm.weight":
            assert parts and all(ref() is None for ref in parts)
        value = original(plan, reader, device)
        if plan.transform == "half_to_float":
            parts.append(weakref.ref(value))
        return value

    monkeypatch.setattr(gguf_weights, "_array", read)
    weights = gguf_weights.load_gguf(path, tp=(3, 8))
    try:
        assert weights.nbytes() == gguf_weights.weight_byte_plan(path, (3, 8))["resident_bytes"]
        assert all(layer.moe.router.dtype == torch.float32 for layer in weights.layers)
        assert all(layer.moe.router.float().data_ptr() == layer.moe.router.data_ptr() for layer in weights.layers)
        assert all(ref() is None for ref in parts)
    finally:
        weights.meta["ple_tables"].close(lambda: torch.cuda.synchronize())


@pytest.mark.parametrize("rows", [1, 3, 6, 18])
@pytest.mark.parametrize("independent", [False, True])
def test_router_float_storage_preserves_logits_and_captured_dynamic_inputs(rows, independent):
    torch.manual_seed(9462)
    half = torch.randn((513, 2560), device="cuda", dtype=torch.float16) * .02
    stored = half.float()
    x = torch.randn((rows, 2560), device="cuda", dtype=torch.float16)
    expected = moe.router(x, half, independent_rows=independent)
    exact(moe.router(x, stored, independent_rows=independent), expected)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = moe.router(x, stored, independent_rows=independent)
    x.mul_(.5)
    graph.replay()
    torch.cuda.synchronize()
    exact(actual, moe.router(x, half, independent_rows=independent))


@pytest.mark.parametrize("cache", ["fp16", "int8"])
def test_router_float_storage_preserves_full_forward_and_cached_prefix_state(cache, monkeypatch):
    from tensorfold.families.qwen4_exp.cuda.forward import commit, forward
    from tensorfold.families.qwen4_exp.cuda.state import Buffers, State

    monkeypatch.setattr(family_ops, "_NATIVE_FAMILY", True)
    monkeypatch.setattr(family_ops, "_NATIVE_ATTENTION", False)
    weights = _small_weights(include_ple=True)
    half = [layer.moe.router for layer in weights.layers]
    stored = [value.float() for value in half]
    states = [State(weights, 32, 8, cache) for _ in range(2)]
    prompt, decode = Buffers(weights, 8, 32, prefill=True), Buffers(weights, 6, 32, moe_prefill=True)

    def use(routers):
        for layer, router in zip(weights.layers, routers):
            layer.moe.router = router

    def compare(a, b):
        assert a.pos == b.pos and a.cur == b.cur
        for index, parity in enumerate(a.cur):
            exact(a.rec[parity, index], b.rec[b.cur[index], index])
        for name in ("conv", "ple_tail"):
            exact(getattr(a, name), getattr(b, name))
        for ka, kb in zip(a.kc, b.kc):
            for name in ("k", "v", "ks", "vs"):
                exact(getattr(ka, name)[:a.pos], getattr(kb, name)[:b.pos])
        for name in ("ikc", "pooled"):
            end = a.pos if name == "ikc" else a.pos // a.ratio
            for va, vb in zip(getattr(a, name), getattr(b, name)):
                exact(va[:end], vb[:end])
        np.testing.assert_array_equal(a.ple_history, b.ple_history)

    logits = []
    for routers, state in zip((half, stored), states):
        use(routers)
        logits.append(forward(weights, state, prompt, [3, 7, 5, 9, 2]).clone())
        commit(weights, state, prompt, 5, 5)
    exact(*logits)
    compare(*states)
    tails = [State(weights, 32, 8, cache) for _ in range(2)]
    for source, tail in zip(states, tails):
        tail.copy_prefix(source, 5, 0)
        tail.restore(source.snapshot())
    for buffers, tokens, keep in ((prompt, [4], 1), (decode, [11, 12, 13], 2)):
        logits = []
        for routers, state in zip((half, stored), tails):
            use(routers)
            logits.append(forward(weights, state, buffers, tokens).clone())
            commit(weights, state, buffers, len(tokens), keep)
        exact(*logits)
        compare(*tails)

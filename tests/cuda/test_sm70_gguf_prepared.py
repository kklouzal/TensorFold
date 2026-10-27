"""Owned M32 staging removes copies without changing fixed-M32 projection arithmetic."""

from dataclasses import replace

import numpy as np
import pytest
import torch

from test_flashnext_sm70_family import _small_weights
from tensorfold.cuda.sm70 import family_ops
from tensorfold.cuda.sm70.gguf_ops import dense32
from tensorfold.families.qwen4_exp.cuda import padded_linears
from tensorfold.families.qwen4_exp.cuda.gguf_weights import GGUFDense, GGUFConcat

pytestmark = pytest.mark.skipif(torch.cuda.get_device_capability() != (7, 0), reason="SM70 M32 owned staging")


def exact(actual, expected):
    kind = torch.int16 if actual.dtype == torch.float16 else torch.int32
    torch.testing.assert_close(actual.contiguous().view(kind), expected.contiguous().view(kind), rtol=0, atol=0)


@pytest.mark.parametrize("rows", [1, 3, 18, 32])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("kind", ["dense", "hc", "qkv", "gdn"])
def test_prepared_full_storage_preserves_original_part_bits_and_guard_regions(rows, dtype, kind):
    torch.manual_seed(3119)
    columns, widths = {"dense": (320, [10240]), "hc": (10240, [320, 4]),
                       "qkv": (2560, [1536, 256, 256, 512, 128]),
                       "gdn": (2560, [1280, 768, 6, 6])}[kind]
    parts = [GGUFDense(torch.randn((width, columns), device="cuda", dtype=torch.float16) * .02)
             for width in widths]
    linear = parts[0] if kind == "dense" else GGUFConcat(parts)
    live = torch.randn((rows, columns), device="cuda", dtype=torch.float16)
    expected = linear.matmul_reference(live, f32=dtype == torch.float32)
    prepared = torch.full((32, columns), torch.nan, device="cuda", dtype=torch.float16)
    prepared[:rows].copy_(live)
    guard = 512 // torch.empty((), dtype=dtype).element_size()
    storage = torch.full((guard + 32 * linear.n + guard,), -739, dtype=dtype, device="cuda")
    output = storage[guard:-guard].view(32, linear.n)
    assert prepared.data_ptr() % 512 == output.data_ptr() % 512 == 0
    assert linear.matmul_prepared(prepared, output) is output
    exact(output[:rows], expected)
    assert torch.all(storage[:guard] == -739) and torch.all(storage[-guard:] == -739)
    # Stale/nonfinite padding must never participate in a live row's dot product.
    prepared[rows:].fill_(203)
    linear.matmul_prepared(prepared, output)
    exact(output[:rows], expected)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_prepared_dense_replays_dynamic_inputs_without_replacing_owned_storage(dtype):
    torch.manual_seed(3120)
    linear = GGUFDense(torch.randn((80, 256), dtype=torch.float16, device="cuda") * .02)
    x = torch.zeros((32, 256), dtype=torch.float16, device="cuda")
    output = torch.empty((32, 80), dtype=dtype, device="cuda")
    linear.matmul_prepared(x, output)
    before = (x.data_ptr(), output.data_ptr())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        linear.matmul_prepared(x, output)
    for rows in (3, 1, 18):
        x[:rows].normal_()
        graph.replay()
        torch.cuda.synchronize()
        exact(output[:rows], linear.matmul_reference(x[:rows], f32=dtype == torch.float32))
        assert before == (x.data_ptr(), output.data_ptr())


def test_prepared_rejects_overlap_and_wrong_rows_before_native_launch():
    linear = GGUFDense(torch.ones((256, 256), dtype=torch.float16, device="cuda"))
    x = torch.zeros((32, 256), dtype=torch.float16, device="cuda")
    with pytest.raises(ValueError, match="overlap"):
        linear.matmul_prepared(x, x)
    with pytest.raises(ValueError, match="M32"):
        linear.matmul_prepared(x[:3], torch.empty((3, 256), dtype=x.dtype, device=x.device))
    linear.implementation = "wmma"
    assert not linear.can_matmul_prepared(torch.float16)


@pytest.mark.parametrize("force_f32", [False, True])
def test_generic_full32_misaligned_operands_retain_reference_and_force_f32_boundary(force_f32):
    torch.manual_seed(3121)
    linear = GGUFDense(torch.randn((81, 256), dtype=torch.float16, device="cuda") * .02)
    x = torch.randn((32 * 256 + 1,), dtype=torch.float16, device="cuda")[1:].view(32, 256)
    output = torch.full((32 * 81 + 2,), -97, dtype=torch.float16, device="cuda")
    dense32(x, linear.weight, output[1:-1].view(32, 81), force_f32=force_f32)
    expected = linear.matmul_reference(x, f32=force_f32).half()
    exact(output[1:-1].view(32, 81), expected)
    assert output[0] == output[-1] == -97


def dense_fixture(w):
    def replace(value):
        return GGUFDense(value.weight)
    w.head = replace(w.head)
    for hc in [w.mixer, *[part for layer in w.layers for part in (layer.attn_hc, layer.mlp_hc)]]:
        hc.down, hc.up = replace(hc.down), replace(hc.up)
    for layer in w.layers:
        branch = layer.gdn if layer.linear else layer.attn
        branch.proj = replace(branch.proj)
        if layer.linear:
            branch.out = replace(branch.out)
        else:
            branch.o = replace(branch.o)
        if layer.ple is not None:
            layer.ple.key, layer.ple.value = replace(layer.ple.key), replace(layer.ple.value)
    return w


@pytest.mark.parametrize("cache", ["fp16", "int8"])
def test_prepared_existing_forward_partial_commit_prefix_copy_and_full_state(cache, monkeypatch):
    from tensorfold.families.qwen4_exp.cuda.forward import commit, forward
    from tensorfold.families.qwen4_exp.cuda.state import Buffers, State

    monkeypatch.setattr(family_ops, "_NATIVE_FAMILY", True)
    monkeypatch.setattr(family_ops, "_NATIVE_ATTENTION", False)
    w = dense_fixture(_small_weights(include_ple=True))
    # Multiple layers make the logical projection bank noncontiguous across
    # layers, while each matrix and every conv-commit row remains contiguous.
    w.layers.append(replace(w.layers[0], index=2, ple=None))
    w.cfg.layers = 3
    w.cfg.layer_types.append("linear")
    states = [State(w, 32, 8, cache) for _ in range(2)]
    monkeypatch.setattr(padded_linears, "_ENABLED", False)
    original = Buffers(w, 8, 32, moe_prefill=True)
    monkeypatch.setattr(padded_linears, "_ENABLED", True)
    prepared = Buffers(w, 8, 32, moe_prefill=True)
    assert original.padded_linears is None and prepared.padded_linears is not None
    assert Buffers(w, 8, 32, prefill=True).padded_linears is None

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

    for tokens, keep in (([3, 7, 5], 3), ([11, 12, 13], 2), ([19], 1), ([21, 22, 23], 1)):
        outputs = []
        for state, buffers in zip(states, (original, prepared)):
            outputs.append(forward(w, state, buffers, tokens).clone())
            commit(w, state, buffers, len(tokens), keep)
        exact(*outputs)
        exact(original.streams[:len(tokens)], prepared.streams[:len(tokens)])
        compare(*states)
    copies = [State(w, 32, 8, cache) for _ in range(2)]
    for state, copy in zip(states, copies):
        copy.copy_prefix(state, state.pos, 0)
        copy.restore(state.snapshot())
    compare(*copies)


@pytest.mark.parametrize("cache", ["fp16", "int8"])
@pytest.mark.parametrize("mtp", [False, True])
def test_prepared_bound_six_lane_capture_preserves_valid_rows_state_and_mtp(cache, mtp, monkeypatch):
    from test_flashnext_sm70_bound_graphs import _qualify_mtp, _qualify_six_lanes
    from test_flashnext_sm70_mtp import _weights_with_stock_mtp_protocol

    monkeypatch.setattr(family_ops, "_NATIVE_FAMILY", True)
    monkeypatch.setattr(family_ops, "_NATIVE_ATTENTION", True)
    monkeypatch.setattr(padded_linears, "_ENABLED", True)
    w = dense_fixture(_weights_with_stock_mtp_protocol() if mtp else _small_weights(include_ple=True))
    _qualify_six_lanes(w, cache)
    if mtp:
        _qualify_mtp(w, cache)

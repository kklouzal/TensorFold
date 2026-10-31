"""Stock MTP FP16 arithmetic on real SM70, including the shared cache/graph path."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tensorfold.families.qwen4_exp.cuda.sm70_mtp_weights import Dense, DenseExperts

pytestmark = pytest.mark.skipif(torch.cuda.get_device_capability() != (7, 0), reason="SM70 stock MTP")


def test_native_dense_fp32_accumulation_output_contract_and_dynamic_graph():
    torch.manual_seed(376)
    weight = torch.randn((97, 128), device="cuda", dtype=torch.float16) * 0.1
    x = torch.randn((35, 128), device="cuda", dtype=torch.float16) * 0.1
    dense = Dense(weight)
    output = dense.matmul(x, f32=True)
    reference = x.cpu().double() @ weight.cpu().double().T
    torch.testing.assert_close(output.cpu().double(), reference, atol=3e-7, rtol=4e-5)
    singles = torch.cat([dense.matmul(row[None], f32=True) for row in x])
    torch.testing.assert_close(singles, output, atol=0, rtol=0)
    torch.testing.assert_close(dense.partials(x)[0], output, atol=0, rtol=0)
    half = torch.empty_like(output, dtype=torch.float16)
    assert dense(x, out=half) is half
    torch.testing.assert_close(half, output.half(), atol=0, rtol=0)
    with pytest.raises(ValueError, match="output"):
        dense.matmul(x, out=half, f32=True)
    replay_output = torch.empty_like(output)
    dense.matmul(x, out=replay_output, f32=True)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        dense.matmul(x, out=replay_output, f32=True)
    x.mul_(0.7)
    expected = dense.matmul(x, f32=True)
    graph.replay()
    torch.testing.assert_close(replay_output, expected, atol=0, rtol=0)


def test_native_experts_fp16_storage_fp32_accumulation_and_dynamic_routes_graph():
    torch.manual_seed(914)
    gate, up = [torch.randn((4, 80, 128), device="cuda", dtype=torch.float16) * 0.1 for _ in range(2)]
    down = torch.randn((4, 128, 80), device="cuda", dtype=torch.float16) * 0.1
    x = torch.randn((3, 128), device="cuda", dtype=torch.float16) * 0.1
    picks = torch.tensor([[0, 3, 1], [2, 0, 3], [1, 1, 2]], device="cuda", dtype=torch.int32)
    bank = DenseExperts(gate, up, down, tile_pairs=4)
    output = bank.forward(x, picks)
    g, u, d, inputs = [v.cpu().double().numpy() for v in (gate, up, down, x)]
    reference = np.empty(output.shape, dtype=np.float64)
    for row in range(3):
        for slot, expert in enumerate(picks.cpu().numpy()[row]):
            # The model stores each projection and SiLU activation in FP16.
            gv = (g[expert] @ inputs[row]).astype(np.float16).astype(np.float64)
            uv = (u[expert] @ inputs[row]).astype(np.float16)
            activation = ((gv / (1 + np.exp(-gv))).astype(np.float16) * uv).astype(np.float16)
            reference[row, slot] = d[expert] @ activation.astype(np.float64)
    np.testing.assert_allclose(output.cpu().numpy(), reference, atol=4e-6, rtol=0.004)
    half = torch.empty_like(output, dtype=torch.float16)
    assert bank.forward(x, picks, out=half) is half
    torch.testing.assert_close(half, output.half(), atol=0, rtol=0)
    replay_output = torch.empty_like(output)
    bank.forward(x, picks, out=replay_output)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        bank.forward(x, picks, out=replay_output)
    picks.copy_((picks + 1) % 4)
    x.mul_(0.7)
    expected = bank.forward(x, picks)
    graph.replay()
    torch.testing.assert_close(replay_output, expected, atol=0, rtol=0)


def _weights_with_stock_mtp_protocol():
    from test_flashnext_sm70_family import _small_weights
    from tensorfold.families.qwen4_exp.cuda.weight_types import MTPW, MoEW

    w = _small_weights(True)
    c = w.cfg

    def linear(previous):
        return Dense(previous.weight)

    def hc(previous):
        return replace(previous, down=linear(previous.down), up=linear(previous.up))

    layer = w.layers[1]
    attention = replace(layer.attn, proj=linear(layer.attn.proj), o=linear(layer.attn.o))
    shape = (c.experts + 1, c.moe_width, c.hidden)
    gate, up = [torch.randn(shape, device=w.device, dtype=torch.float16) * 0.015 for _ in range(2)]
    down = torch.randn((c.experts + 1, c.hidden, c.moe_width), device=w.device, dtype=torch.float16) * 0.015
    layer = replace(layer, index=c.layers, attn=attention, attn_hc=hc(layer.attn_hc),
                    mlp_hc=hc(layer.mlp_hc), moe=MoEW(layer.moe.router, DenseExperts(gate, up, down)))
    fusion = [Dense(torch.randn((c.hidden, c.hidden), device=w.device, dtype=torch.float16) * 0.03)
              for _ in range(2)]
    w.mtp = MTPW(torch.ones(c.hidden, device=w.device), torch.ones(c.streams * c.hidden, device=w.device),
                 *fusion, layer, hc(w.mixer))
    return w


@pytest.mark.parametrize("cache", ["fp16", "int8"])
def test_shared_stock_mtp_chunk_prefix_copy_and_graph_use_advancing_own_cache(cache):
    from tensorfold.families.qwen4_exp.cuda.forward import forward
    from tensorfold.families.qwen4_exp.cuda.graphs import Graphs
    from tensorfold.families.qwen4_exp.cuda.mtp import mtp_forward
    from tensorfold.families.qwen4_exp.cuda.state import Buffers, State

    w = _weights_with_stock_mtp_protocol()
    serial, batched = [State(w, 24, 12, cache) for _ in range(2)]
    buf = Buffers(w, 12, 24)
    tokens = [4, 17, 9, 31, 16, 25, 13, 15, 7]
    # Consume actual final residual streams from the target's shared forward.
    forward(w, batched, buf, tokens)
    streams = buf.streams[:len(tokens)].clone()
    whole = mtp_forward(w, batched, buf, tokens, streams, last_only=False).clone()
    batched.set_mtp_len(len(tokens))
    singles = []
    for row, token in enumerate(tokens):
        singles.append(mtp_forward(w, serial, buf, [token], streams[row:row + 1]).clone())
        serial.set_mtp_len(row + 1)
    torch.testing.assert_close(torch.cat(singles), whole, atol=0.001, rtol=0)
    for name in ("k", "v", "ks", "vs"):
        torch.testing.assert_close(getattr(serial.mtp_kc, name), getattr(batched.mtp_kc, name), atol=0, rtol=0)
    graph_state, eager_state = [State(w, 24, 12, cache) for _ in range(2)]
    for state in (graph_state, eager_state):
        state.copy_prefix(batched, 0, batched.mtp_len)
        state.set_mtp_len(batched.mtp_len)
    graph_buf, eager_buf = Buffers(w, 12, 24), Buffers(w, 12, 24)
    graph = Graphs(SimpleNamespace(w=w, st=graph_state, buf=buf, mbuf=graph_buf), max_rows=2)
    for step, next_tokens in enumerate(([71, 33], [18, 91], [2, 8])):
        current_streams = streams[step:step + 2]
        expected = mtp_forward(w, eager_state, eager_buf, next_tokens, current_streams).clone()
        actual = graph.mtp_forward(next_tokens, current_streams).clone()
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        for state in (graph_state, eager_state):
            state.set_mtp_len(state.mtp_len + len(next_tokens))
        for name in ("k", "v", "ks", "vs"):
            torch.testing.assert_close(getattr(graph_state.mtp_kc, name), getattr(eager_state.mtp_kc, name),
                                       atol=0, rtol=0)
        torch.testing.assert_close(graph_state.mtp_pooled, eager_state.mtp_pooled, atol=0, rtol=0)
    assert graph.captures == 1

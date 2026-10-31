"""Actual Volta component gates; these do not qualify Flash Next model serving."""

import numpy as np
import pytest
import torch

from tensorfold.cuda.sm70 import SM70Provider

pytestmark = pytest.mark.skipif(torch.cuda.get_device_capability() != (7, 0), reason="SM70-only provider")


@pytest.fixture(scope="module")
def provider():
    result = SM70Provider()
    result.prepare()
    return result


@pytest.mark.parametrize("rows,width", [(1, 80), (3, 640), (2, 2560), (4, 10240), (0, 128), (2, 131)])
def test_native_rms_norm_against_float64(provider, rows, width):
    rng = np.random.default_rng(321)
    data = rng.normal(size=(rows, width)).astype(np.float16)
    gamma = rng.normal(size=width).astype(np.float32)
    x = torch.tensor(data, device="cuda")
    result = provider.rms_norm(x, torch.tensor(gamma, device="cuda"), 1e-6)
    expected = data.astype(np.float64) / np.sqrt(np.mean(data.astype(np.float64) ** 2, axis=1, keepdims=True) + 1e-6)
    expected = (expected * gamma).astype(np.float16)
    np.testing.assert_allclose(result.cpu().numpy(), expected, rtol=0.002, atol=0.002)


@pytest.mark.parametrize("rows,heads,width,rotary", [(1, 4, 256, 64), (3, 2, 128, 64), (0, 4, 256, 64)])
def test_native_rope_half_pairing_and_unchanged_tail(provider, rows, heads, width, rotary):
    rng = np.random.default_rng(117)
    data = rng.normal(size=(rows, heads, width)).astype(np.float16)
    freq = (1e7 ** (-np.arange(rotary // 2, dtype=np.float64) * 2 / rotary)).astype(np.float32)
    positions = np.arange(rows, dtype=np.int64) + 173
    result = provider.rope(torch.tensor(data, device="cuda"), torch.tensor(freq, device="cuda"),
                           torch.tensor(positions, device="cuda")).cpu().numpy()
    expected = data.copy()
    for row in range(rows):
        angle = positions[row] * freq.astype(np.float64)
        a, b = data[row, :, :rotary // 2].astype(np.float64), data[row, :, rotary // 2:rotary].astype(np.float64)
        expected[row, :, :rotary // 2] = a * np.cos(angle) - b * np.sin(angle)
        expected[row, :, rotary // 2:rotary] = b * np.cos(angle) + a * np.sin(angle)
    np.testing.assert_allclose(result[..., :rotary], expected[..., :rotary], rtol=0.003, atol=0.003)
    np.testing.assert_array_equal(result[..., rotary:], data[..., rotary:])


@pytest.mark.parametrize("width", [80, 640, 2560, 10240])
def test_projection_reference_policy(provider, width):
    torch.manual_seed(44)
    x = torch.randn((3, width), device="cuda", dtype=torch.float16) * 0.1
    w = torch.randn((17, width), device="cuda", dtype=torch.float16) * 0.1
    expected = (x.cpu().double() @ w.cpu().double().T).half()
    torch.testing.assert_close(provider.linear(x, w).cpu(), expected, rtol=0.002, atol=0.002)


def test_gqa_causal_attention_against_scalar_head_reference(provider):
    torch.manual_seed(81)
    q = torch.randn((3, 4, 256), device="cuda", dtype=torch.float16)
    k = torch.randn((11, 1, 256), device="cuda", dtype=torch.float16)
    v = torch.randn_like(k)
    expected = torch.empty_like(q, device="cpu")
    for row in range(3):
        for head in range(4):
            scores = k[:9 + row, 0].cpu().double() @ q[row, head].cpu().double() / 16
            expected[row, head] = scores.softmax(0) @ v[:9 + row, 0].cpu().double()
    torch.testing.assert_close(provider.attention(q, k, v, start=8).cpu(), expected, rtol=0.003, atol=0.003)


@pytest.mark.parametrize("rows", [0, 1, 4])
def test_delta_tp8_state_and_chunk_equivalence(provider, rows):
    torch.manual_seed(905)
    q, k = [torch.randn((rows, 2, 128), device="cuda", dtype=torch.float16) * 0.01 for _ in range(2)]
    v = torch.randn((rows, 6, 128), device="cuda", dtype=torch.float16)
    decay = torch.full((rows, 6), 0.9, device="cuda")
    beta = torch.full((rows, 6), 0.4, device="cuda")
    state = torch.randn((6, 128, 128), device="cuda") * 0.01
    saved = state.clone()
    outputs, final = provider.delta_chain(q, k, v, decay, beta, state)
    # Independent CPU FP64 head-by-head oracle, with state[value,key] layout.
    ref = state.cpu().double()
    expected = []
    for row in range(rows):
        out = []
        for head in range(6):
            key, query = k[row, head // 3].cpu().double(), q[row, head // 3].cpu().double()
            ref[head] *= float(decay[row, head])
            delta = (v[row, head].cpu().double() - ref[head] @ key) * float(beta[row, head])
            ref[head] += delta[:, None] * key[None, :]
            out.append((ref[head] @ query).half())
        expected.append(torch.stack(out))
    if rows:
        torch.testing.assert_close(outputs.cpu(), torch.stack(expected), rtol=0.003, atol=0.001)
    torch.testing.assert_close(final.cpu().double(), ref, rtol=2e-5, atol=2e-7)
    torch.testing.assert_close(state, saved, rtol=0, atol=0)
    split = min(1, rows)
    left, middle = provider.delta_chain(q[:split], k[:split], v[:split], decay[:split], beta[:split], state)
    right, end = provider.delta_chain(q[split:], k[split:], v[split:], decay[split:], beta[split:], middle)
    torch.testing.assert_close(torch.cat((left, right)), outputs, rtol=0, atol=0)
    torch.testing.assert_close(end, final, rtol=0, atol=0)


def test_native_current_stream_and_graph_replay(provider):
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        x = torch.ones((2, 640), device="cuda", dtype=torch.float16)
        weight = torch.ones(640, device="cuda")
        provider.rms_norm(x, weight)  # Warm before capture.
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        output = provider.rms_norm(x, weight)
    with torch.cuda.stream(stream):
        x.zero_()
        graph.replay()
    stream.synchronize()
    torch.testing.assert_close(output, torch.zeros_like(output), rtol=0, atol=0)


def test_native_rejects_invalid_inputs_before_launch(provider):
    x = torch.ones((2, 80), device="cuda", dtype=torch.float16)
    with pytest.raises(RuntimeError, match="FP32"):
        provider.rms_norm(x, torch.ones(80, device="cuda", dtype=torch.float16))
    with pytest.raises(RuntimeError, match="contiguous"):
        provider.rms_norm(x[:, ::2], torch.ones(40, device="cuda"))
    with pytest.raises(ValueError, match="epsilon"):
        provider.rms_norm(x, torch.ones(80, device="cuda"), float("nan"))

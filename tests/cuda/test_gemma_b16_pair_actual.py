"""Root-only B01 actual-weight bit/partition/graph oracles; no speed claim.

Set TENSORFOLD_B01_MODEL_DIR to the immutable pinned llmfan46 snapshot. Only
four BF16 projection pairs are read; no model/expert loading or weight copies
beyond the original CPU->CUDA transfer. Independent b16 calls remain oracles.
"""
import hashlib
import json
import os
from pathlib import Path

import pytest
import torch
from safetensors import safe_open

from tensorfold.families.qwen3_5.cuda.b16 import matmul, matmul_pair, prompt, prompt_pair


def assert_bits(actual, expected):
    assert actual.dtype == expected.dtype == torch.bfloat16
    assert actual.shape == expected.shape
    assert torch.isfinite(actual).all() and torch.isfinite(expected).all()
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))


@pytest.fixture(params=[('kv', 0), ('kv', 28), ('gate_up', 0), ('gate_up', 29)])
def actual_pair(request, record_property):
    root = os.environ.get('TENSORFOLD_B01_MODEL_DIR')
    if root is None:
        pytest.skip('explicit immutable actual-weight snapshot required')
    root = Path(root)
    expected = '56ed14f6dd662fcefcc6b99b7f25c68cb4ed3978af9abf2ee179fa859877d37e'
    assert hashlib.sha256((root / 'config.json').read_bytes()).hexdigest() == expected
    raw = json.loads((root / 'config.json').read_bytes())
    index = json.loads((root / 'model.safetensors.index.json').read_bytes())['weight_map']
    role, layer = request.param
    names = (('self_attn.k_proj', 'self_attn.v_proj') if role == 'kv'
             else ('mlp.gate_proj', 'mlp.up_proj'))
    if role == 'kv':
        assert raw['text_config']['layer_types'][layer] == 'sliding_attention'
    n = 2048 if role == 'kv' else 2112
    weights = []
    for suffix in names:
        name = f'model.language_model.layers.{layer}.{suffix}.weight'
        with safe_open(str(root / index[name]), framework='pt', device='cpu') as source:
            value = source.get_tensor(name)
            assert value.dtype == torch.bfloat16 and value.shape == (n, 2816) and value.is_contiguous()
            weights.append(value.to('cuda'))
        value = None
    record_property('actual_projection_role', f'{role}:layer{layer}:N{n}:K2816')
    record_property('source_revision', root.name)
    return weights


def test_actual_b16_pair_decode_prompt_partitions_and_graph(actual_pair):
    w0, w1 = actual_pair
    generator = torch.Generator(device='cuda').manual_seed(613)
    x = torch.empty((1024, 2816), device='cuda', dtype=torch.bfloat16).uniform_(-.5, .5, generator=generator)
    x[0].zero_()
    x[1].mul_(32)  # finite adverse magnitude, no change to reference precision
    pointers = w0.data_ptr(), w1.data_ptr()
    for rows in range(1, 9):
        pair = matmul_pair(x[:rows], w0, w1)
        assert_bits(pair[0], matmul(x[:rows], w0))
        assert_bits(pair[1], matmul(x[:rows], w1))
        assert_bits(pair[0], torch.cat([matmul(x[r:r+1], w0) for r in range(rows)]))
        assert_bits(pair[1], torch.cat([matmul(x[r:r+1], w1) for r in range(rows)]))
        assert pair[0].data_ptr() != pair[1].data_ptr()
    for rows in (1, 3, 17, 33, 1024):
        pair = prompt_pair(x[:rows], w0, w1)
        assert_bits(pair[0], prompt(x[:rows], w0))
        assert_bits(pair[1], prompt(x[:rows], w1))
    full = prompt_pair(x, w0, w1)
    chunks = [prompt_pair(x[a:b], w0, w1) for a, b in ((0, 1), (1, 4), (4, 21), (21, 54), (54, 1024))]
    assert_bits(torch.cat([part[0] for part in chunks]), full[0])
    assert_bits(torch.cat([part[1] for part in chunks]), full[1])
    # Extensions are prepared/warmed by the calls above. Capture the actual
    # paired operator using the supported public graph context; no retry on a
    # capture/native failure. The original weights remain borrowed/immutable.
    for rows, paired, independent in ((1, matmul_pair, matmul), (33, prompt_pair, prompt)):
        static = x[:rows].clone()
        graph = torch.cuda.CUDAGraph()
        torch.cuda.synchronize()
        with torch.cuda.graph(graph):
            held = paired(static, w0, w1)
        graph.replay()
        assert_bits(held[0], independent(static, w0))
        assert_bits(held[1], independent(static, w1))
        static.mul_(.75)
        graph.replay()
        assert_bits(held[0], independent(static, w0))
        assert_bits(held[1], independent(static, w1))
        assert held[0].data_ptr() != held[1].data_ptr()
    torch.cuda.synchronize()
    assert (w0.data_ptr(), w1.data_ptr()) == pointers

"""Exact owned Half/11-slot/D2560 reduction against the unchanged Torch oracle."""
from unittest.mock import patch

import pytest
import torch

from tensorfold.cuda.sm70 import family_ops as ops

pytestmark = pytest.mark.skipif(not torch.cuda.is_available()
    or torch.cuda.get_device_capability() != (7, 0), reason='requires actual SM70')


@pytest.fixture(scope='module', autouse=True)
def resident_partial():
    with patch.object(ops, '_PRIVATE_HALF_PARTIAL', 'native-half-v1'):
        module = ops.prepare_half_moe_partial(torch.device('cuda', torch.cuda.current_device()))
    assert module is not None
    return module


def native(y, weights, out, rows):
    # Exercise only the private Half policy, never the whole-family switch.
    with patch.object(ops, '_PRIVATE_HALF_PARTIAL', 'native-half-v1'), patch.object(ops, '_NATIVE_FAMILY', False):
        return ops.moe_partial(y, weights, out, rows)


def same_bits(actual, expected):
    a, b = actual.view(torch.int32), expected.view(torch.int32)
    different = (a != b).nonzero()
    first = None if different.numel() == 0 else different[0].tolist()
    assert first is None, f'FP32 storage mismatch at {first}'


@pytest.fixture(scope='module')
def finite_inputs():
    generator = torch.Generator(device='cuda').manual_seed(19371)
    y = torch.randn((257, 11, 2560), generator=generator, device='cuda', dtype=torch.float16)
    weights = torch.randn((257, 11), generator=generator, device='cuda')
    # Signed weights and high/low magnitudes exercise ordered cancellation.
    weights[:, 0::2] *= 32
    weights[:, 1::2] *= -32
    return y, weights


@pytest.mark.parametrize('rows', range(1, 257))
def test_every_live_prefix_bitwise_and_inactive_canary(rows, finite_inputs):
    y, weights = finite_inputs
    expected = torch.full((257, 2560), -719., device='cuda')
    actual = expected.clone()
    before = actual._version
    ops._reference['moe_partial'](y, weights, expected, rows)
    native(y, weights, actual, rows)
    same_bits(actual, expected)
    assert actual._version == before + 1


@pytest.mark.parametrize('weight_kind', ['unit', 'signed-zero', 'cancellation', 'extreme', 'nonfinite'])
def test_every_half_bit_pattern_and_fp32_weight_boundaries(weight_kind):
    words = torch.arange(65536, device='cuda', dtype=torch.int32).to(torch.int16)
    y = words.repeat(2)[:4*11*2560].view(torch.float16).reshape(4, 11, 2560)
    weights = torch.ones((4, 11), device='cuda')
    if weight_kind == 'signed-zero':
        weights[:, ::2] = -0.
        weights[:, 1::2] = 0.
    elif weight_kind == 'cancellation':
        weights[:, 1::2] = -1.
    elif weight_kind == 'extreme':
        weights[:, 0] = torch.finfo(torch.float32).max
        weights[:, 1] = torch.finfo(torch.float32).tiny
        weights[:, 2] = torch.nextafter(torch.zeros((), device='cuda'), torch.ones((), device='cuda'))
    elif weight_kind == 'nonfinite':
        weights[:, 0] = float('inf')
        weights[:, 1] = -float('inf')
        weights[:, 2] = float('nan')
    expected, actual = [torch.empty((4, 2560), device='cuda') for _ in range(2)]
    ops._reference['moe_partial'](y, weights, expected, 4)
    native(y, weights, actual, 4)
    same_bits(actual, expected)


def test_nondefault_stream_and_graph_replay_keep_live_inputs(finite_inputs):
    y, weights = [t[:7].clone() for t in finite_inputs]
    expected, actual = [torch.empty((7, 2560), device='cuda') for _ in range(2)]
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        ops._reference['moe_partial'](y, weights, expected, 7)
        native(y, weights, actual, 7)
    torch.cuda.current_stream().wait_stream(stream)
    same_bits(actual, expected)
    # Warm the exact callable/module before capture; no lazy loading in capture.
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        native(y, weights, actual, 7)
    version = actual._version
    for factor in [0., -1., .125]:
        weights.fill_(factor)
        ops._reference['moe_partial'](y, weights, expected, 7)
        graph.replay()
        same_bits(actual, expected)
        assert actual._version == version


@pytest.mark.parametrize('inference', [False, True])
@pytest.mark.parametrize('rows', [0, 1])
def test_empty_and_inference_mutation_contract(inference, rows):
    with torch.inference_mode(inference):
        y = torch.ones((1, 11, 2560), dtype=torch.float16, device='cuda')
        weights = torch.ones((1, 11), device='cuda')
        expected = torch.full((1, 2560), 9., device='cuda')
        actual = expected.clone()
    before = None if inference else actual._version
    with torch.inference_mode(inference):
        ops._reference['moe_partial'](y, weights, expected, rows)
        native(y, weights, actual, rows)
    same_bits(actual, expected)
    if inference:
        assert actual.is_inference()
    else:
        assert actual._version == before + 1


@pytest.mark.parametrize('fault', ['weights-dtype', 'out-dtype', 'negative-rows', 'too-many-rows', 'slots', 'width', 'out-cpu'])
def test_abi_refuses_invalid_operands_before_output_mutation(fault):
    y = torch.ones((2, 11, 2560), dtype=torch.float16, device='cuda')
    weights = torch.ones((2, 11), device='cuda')
    out = torch.full((2, 2560), 9., device='cuda')
    rows = 2
    if fault == 'weights-dtype':
        weights = weights.half()
    elif fault == 'out-dtype':
        out = out.half()
    elif fault == 'negative-rows':
        rows = -1
    elif fault == 'too-many-rows':
        rows = 3
    elif fault == 'slots':
        weights = weights[:, :10].contiguous()
    elif fault == 'width':
        out = torch.ones((2, 2559), device='cuda')
    else:
        out = out.cpu()
    before, saved = out._version, out.clone()
    with pytest.raises(RuntimeError):
        native(y, weights, out, rows)
    same_bits(out.float(), saved.float())
    assert out._version == before


@pytest.mark.parametrize('domain', ['values-stride', 'weights-stride', 'out-stride', 'alias-values',
    'alias-weights', 'grad', 'subclass', 'negative-view', 'inference-outside-mode'])
def test_generic_domains_keep_reference_temporary_semantics(domain):
    y = torch.randn((2, 11, 2560), dtype=torch.float16, device='cuda')
    weights = torch.randn((2, 11), device='cuda')
    out = torch.empty((2, 2560), device='cuda')
    if domain == 'values-stride':
        y = torch.randn((2, 11, 5120), dtype=torch.float16, device='cuda')[:, :, ::2]
    elif domain == 'weights-stride':
        weights = torch.randn((2, 22), device='cuda')[:, ::2]
    elif domain == 'out-stride':
        out = torch.empty((2, 5120), device='cuda')[:, ::2]
    elif domain == 'alias-values':
        out = y.view(torch.float32).reshape(-1)[:5120].view(2, 2560)
    elif domain == 'alias-weights':
        storage = torch.randn(5120, device='cuda')
        out, weights = storage.view(2, 2560), storage[:22].view(2, 11)
    elif domain == 'grad':
        y.requires_grad_(True)
    elif domain == 'subclass':
        class Other(torch.Tensor):
            pass
        y = y.as_subclass(Other)
    elif domain == 'negative-view':
        y = torch._neg_view(y)
    else:
        with torch.inference_mode():
            out = torch.empty((2, 2560), device='cuda')
    expected = torch.empty((2, 2560), device='cuda')
    ops._reference['moe_partial'](y, weights, expected, 2)
    # Inference outputs outside inference mode retain Torch's refusal.
    if domain == 'inference-outside-mode':
        with pytest.raises(RuntimeError):
            native(y, weights, out, 2)
    else:
        native(y, weights, out, 2)
        same_bits(out, expected)
        if domain == 'grad':
            assert out.requires_grad


def test_actual_prefill4096_prefix_tail_and_shared_view_version():
    generator = torch.Generator(device='cuda').manual_seed(4096112560)
    y = torch.randn((4097, 11, 2560), dtype=torch.float16, device='cuda', generator=generator)
    weights = torch.randn((4097, 11), device='cuda', generator=generator)
    weights[:, 1::2].neg_()
    expected_base = torch.full((4098, 2560), -719., device='cuda')
    actual_base = expected_base.clone()
    expected, actual = expected_base[1:], actual_base[1:]
    before = actual_base._version
    ops._reference['moe_partial'](y, weights, expected, 4096)
    native(y, weights, actual, 4096)
    same_bits(actual_base, expected_base)
    assert actual._version == actual_base._version == before + 1

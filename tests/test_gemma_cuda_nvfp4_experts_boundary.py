"""NativeCT Gemma metadata/packing boundaries without importing CUDA runtimes."""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1] / 'src/tensorfold'


def _definitions(path, names, namespace):
    nodes = [ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)]
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names:
            node.decorator_list = []
            nodes.append(node)
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), str(path), 'exec'), namespace)
    return namespace


class Tensor:
    def __init__(self, shape, *, dtype='bf16', device='cuda:0'):
        self.shape, self.dtype, self.device = shape, dtype, device
        self.ndim = len(shape)

    def numel(self):
        from math import prod

        return prod(self.shape)

    def element_size(self):
        return 1 if self.dtype == 'u8' else 4


class Bank:
    def __init__(self, swiglu, *, direct=False):
        self.swiglu, self.direct_input = swiglu, direct
        self.count, self.width, self.dims, self.device = 128, 704, 2816, 'cuda:0'

    def device_tensors(self):
        return Tensor((128, 2, 1115136), dtype='u8'), Tensor((128, 1115136), dtype='u8'), Tensor((768,), dtype='f32')


def _namespace():
    calls = []

    def allocate(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError('invalid Gemma NVFP4 boundary reached an allocation')

    torch = SimpleNamespace(Tensor=Tensor, bfloat16='bf16', int32='i32', float32='f32', empty=allocate)
    ns = _definitions(ROOT / 'families/gemma4/cuda/nvfp4_experts.py', {'Experts'},
                      {'torch': torch, 'ExpertsA4': Bank, 'geglu': object()})
    return ns, calls


def test_adapter_rejects_modelopt_direct_scales_and_wrong_activation_contract():
    ns, calls = _namespace()
    for bank in (Bank(ns['geglu'], direct=True), Bank(object()), object()):
        with pytest.raises(ValueError, match='original CT'):
            ns['Experts'](bank)
    assert not calls


def test_native_fields_are_counted_once_without_authoritative_cpu_copy():
    ns, calls = _namespace()
    adapter = ns['Experts'](Bank(ns['geglu']))
    assert adapter.nbytes() == 128 * 3 * 1115136 + 768 * 4
    assert not calls


@pytest.mark.parametrize('x', [None, Tensor((0, 2816)), Tensor((1, 2815)), Tensor((1, 2816), dtype='f32'),
                              Tensor((1, 2816), device='cpu'), Tensor((1, 1, 2816))])
def test_activation_metadata_rejected_before_native_or_cuda_work(x):
    ns, calls = _namespace()
    with pytest.raises(ValueError):
        ns['Experts'](Bank(ns['geglu'])).forward(x, Tensor((1, 8), dtype='i32'), Tensor((1, 8), dtype='f32'))
    assert not calls


@pytest.mark.parametrize('picks', [None, Tensor((1, 8), dtype='i64'), Tensor((2, 8), dtype='i32'),
                                  Tensor((1, 0), dtype='i32'), Tensor((1, 129), dtype='i32'),
                                  Tensor((1, 8), dtype='i32', device='cuda:1')])
def test_pick_metadata_prevents_implicit_integer_narrowing_and_wrong_geometry(picks):
    ns, calls = _namespace()
    with pytest.raises(ValueError):
        ns['Experts'](Bank(ns['geglu'])).forward(Tensor((1, 2816)), picks, Tensor((1, 8), dtype='f32'))
    assert not calls


@pytest.mark.parametrize('weights', [None, Tensor((1, 8), dtype='bf16'), Tensor((1, 7), dtype='f32'),
                                    Tensor((1, 8), dtype='f32', device='cpu')])
def test_routing_weights_retain_fp32_and_exact_row_slot_shape(weights):
    ns, calls = _namespace()
    with pytest.raises(ValueError):
        ns['Experts'](Bank(ns['geglu'])).forward(Tensor((1, 2816)), Tensor((1, 8), dtype='i32'), weights)
    assert not calls


@pytest.mark.parametrize('weight_device,scale_device,weight_dtype,scale_dtype,shape,scale_shape', [
    ('meta', 'meta', 'u8', 'f8', (64, 32), (64, 4)),
    ('cpu', 'cuda', 'u8', 'f8', (64, 32), (64, 4)),
    ('cuda', 'cuda', 'i32', 'f8', (64, 32), (64, 4)),
    ('cpu', 'cpu', 'u8', 'u8', (64, 32), (64, 4)),
    ('cpu', 'cpu', 'u8', 'f8', (63, 32), (63, 4)),
    ('cpu', 'cpu', 'u8', 'f8', (64, 31), (64, 4)),
    ('cpu', 'cpu', 'u8', 'f8', (64, 32), (64, 3)),
])
def test_shared_packer_rejects_bad_devices_dtype_and_partial_tiles_before_kernel(
        weight_device, scale_device, weight_dtype, scale_dtype, shape, scale_shape):
    devices = {name: SimpleNamespace(type=name) for name in ('meta', 'cpu', 'cuda')}
    torch = SimpleNamespace(uint8='u8', float8_e4m3fn='f8')
    ns = _definitions(ROOT / 'cuda/nvfp4/experts_a4.py', {'pack_projection_a4'}, {'torch': torch})
    weight = Tensor(shape, dtype=weight_dtype, device=devices[weight_device])
    scale = Tensor(scale_shape, dtype=scale_dtype, device=devices[scale_device])
    with pytest.raises(ValueError):
        ns['pack_projection_a4'](weight, scale)

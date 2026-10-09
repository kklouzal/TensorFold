"""Gemma routing geometry rejects before GPU work; source fixtures need no MLX."""
from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).parents[1]


def _function(path, symbol, namespace):
    node = next(n for n in ast.parse((ROOT / path).read_bytes()).body
                if isinstance(n, ast.FunctionDef) and n.name == symbol)
    module = ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module='__future__',
        names=[ast.alias(name='annotations')], level=0), node], type_ignores=[]))
    exec(compile(module, '<isolated-Gemma-boundary>', 'exec'), namespace)
    return namespace[symbol]


def _route():
    calls = []

    def kernel(*args, **kwargs):
        calls.append((args, kwargs))
        return 'kernel-result'
    function = _function('src/tensorfold/kernels/gemma/v1/moe.py', 'route',
                         {'_route': kernel, 'MIN_ELEMENTS': 8,
                          'mx': SimpleNamespace(uint32='u32', bfloat16='bf16')})
    return function, calls


@pytest.mark.parametrize('experts,top_k,scale_shape', [
    (0, 1, (0,)), (1, 1, (1,)), (31, 1, (31,)), (33, 1, (33,)),
    (32, 0, (32,)), (32, -1, (32,)), (32, 33, (32,)),
    (32, True, (32,)), (32, 1.0, (32,)),
    (32, 1, (31,)), (32, 1, (33,)), (32, 1, (1, 32)), (32, 1, (32, 1)),
])
def test_route_invalid_geometry_never_calls_kernel(experts, top_k, scale_shape):
    route, calls = _route()
    with pytest.raises(ValueError):
        route(SimpleNamespace(shape=(2, experts)), SimpleNamespace(shape=scale_shape), top_k)
    assert not calls


@pytest.mark.parametrize('rows,experts,top_k', [(0, 32, 1), (1, 32, 1), (3, 32, 4), (2, 64, 8), (1, 128, 128)])
def test_valid_route_keeps_exact_kernel_configuration(rows, experts, top_k):
    route, calls = _route()
    scores, scales = SimpleNamespace(shape=(rows, experts)), SimpleNamespace(shape=(experts,))
    assert route(scores, scales, top_k) == 'kernel-result'
    assert calls == [(((("NE", experts), ("K", top_k)),),
                     dict(inputs=[scores, scales], grid=(32 * rows, 1, 1), threadgroup=(32, 1, 1),
                          output_shapes=[(max(rows * top_k, 8),)] * 2, output_dtypes=['u32', 'bf16']))]


def _constructor():
    tree = ast.parse((ROOT / 'src/tensorfold/kernels/gemma/v1/decode.py').read_bytes())
    owner = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'RowDecode')
    method = next(n for n in owner.body if isinstance(n, ast.FunctionDef) and n.name == '__init__')
    stop = next(i for i, n in enumerate(method.body) if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Attribute) and t.attr == 'window' for t in n.targets))
    method.body = method.body[:stop]
    module = ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module='__future__',
        names=[ast.alias(name='annotations')], level=0), method], type_ignores=[]))
    namespace = {}
    exec(compile(module, '<isolated-Gemma-startup>', 'exec'), namespace)
    return namespace['__init__']


@pytest.mark.parametrize('experts,top_k,scale_shape,weight_experts', [
    (0, 1, (0,), 0), (32, 0, (32,), 32), (32, 33, (32,), 32),
    (32, 4, (31,), 32), (32, 4, (32,), 64),
])
def test_startup_metadata_rejects_before_array_creation(experts, top_k, scale_shape, weight_experts):
    args = SimpleNamespace(enable_moe_block=True, num_experts=experts, top_k_experts=top_k)
    router = SimpleNamespace(proj=SimpleNamespace(weight=SimpleNamespace(shape=(weight_experts, 256))),
                             per_expert_scale=SimpleNamespace(shape=scale_shape))
    model = SimpleNamespace(args=args, model=SimpleNamespace(layers=[SimpleNamespace(router=router)]))
    with pytest.raises(ValueError):
        _constructor()(SimpleNamespace(), model, 'rows')

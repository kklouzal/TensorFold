"""Prepared head-policy integration controls without Torch/device execution."""
import __future__

import ast
from functools import partial
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from tools.gemma4_cuda_launch_qualify import _read

SOURCE = Path(__file__).resolve().parents[1] / 'src/tensorfold/families/gemma4/cuda/head.py'


def maintained():
    calls = []

    class Plain:
        def __init__(self, weight):
            self.weight = weight

        def __call__(self, value):
            calls.append(('original', value, self.weight))
            return calls[-1]

        def nbytes(self):
            return self.weight.bytes

        def prefill(self, value):
            calls.append(('prompt', value, self.weight))
            return calls[-1]

    def multiply(value, weight, *, rows):
        calls.append((rows, value, weight))
        return calls[-1]

    probes = []
    cuda = NS(get_device_capability=lambda device: (probes.append(device), (12, 1))[1])
    scope = {'partial': partial, 'Plain': Plain, 'matmul_rows': multiply,
             'torch': NS(bfloat16='bf16', cuda=cuda)}
    tree = ast.parse(_read(SOURCE, 1 << 20))
    tree.body = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef))]
    exec(compile(tree, str(SOURCE), 'exec', flags=__future__.annotations.compiler_flag), scope)  # noqa: S102 -- maintained AST only
    weight = NS(dtype='bf16', shape=(262144, 2816), device=NS(type='cuda'),
                is_contiguous=lambda: True, bytes=262144 * 2816 * 2)
    return scope, Plain(weight), calls, probes


@pytest.mark.parametrize('rows,expected', [(0, 16), (1, 'original'), (2, 4), (3, 4), (4, 4),
                                          (5, 8), (8, 8), (9, 16), (17, 16), (129, 16)])
def test_prepared_policy_uses_existing_geometry_and_original_zero_singleton_paths(rows, expected):
    scope, incumbent, calls, probes = maintained()
    selected = scope['select_head'](incumbent)
    assert selected.weight is incumbent.weight and selected.incumbent is incumbent
    assert selected.nbytes() == incumbent.nbytes() and len(probes) == 1
    scope['torch'].cuda.get_device_capability = lambda _device: pytest.fail('hot capability discovery')
    value = NS(shape=(rows, 2816))
    assert selected(value) == (expected, value, incumbent.weight)
    assert calls == [(expected, value, incumbent.weight)]
    assert scope['select_head'](selected) is selected


def test_policy_prefill_keeps_the_inherited_plain_interface():
    scope, incumbent, calls, _probes = maintained()
    selected, value = scope['select_head'](incumbent), object()
    assert selected.prefill(value) == ('prompt', value, incumbent.weight)
    assert calls == [('prompt', value, incumbent.weight)]


def test_zero_rows_keep_exact_native_error_without_normalization():
    scope, incumbent, _calls, _probes = maintained()
    primary, calls = RuntimeError('native dimensions refused'), []

    def native(value, weight, *, rows):
        calls.append((value, weight, rows))
        raise primary

    scope['matmul_rows'] = native
    selected, value = scope['select_head'](incumbent), NS(shape=(0, 2816))
    with pytest.raises(RuntimeError) as caught:
        selected(value)
    assert caught.value is primary and calls == [(value, incumbent.weight, 16)]


@pytest.mark.parametrize('reuse_rows', [2, 4, 8, 16])
def test_fixed_comparator_policy_and_metadata_are_preserved(reuse_rows):
    scope, incumbent, calls, _probes = maintained()
    prepared = scope['prepare_head'](incumbent, reuse_rows=reuse_rows)
    value = NS(shape=(5, 2816))
    assert prepared.reuse_rows == reuse_rows and prepared.incumbent is incumbent
    assert prepared(value) == (reuse_rows, value, incumbent.weight)
    assert calls == [(reuse_rows, value, incumbent.weight)]


@pytest.mark.parametrize('change', ['dtype', 'shape', 'host', 'contiguous', 'subclass'])
def test_unqualified_inputs_keep_original_without_new_capability_discovery(change):
    scope, incumbent, _calls, probes = maintained()
    if change == 'dtype':
        incumbent.weight.dtype = 'fp32'
    elif change == 'shape':
        incumbent.weight.shape = (128, 64)
    elif change == 'host':
        incumbent.weight.device.type = 'cpu'
    elif change == 'contiguous':
        incumbent.weight.is_contiguous = lambda: False
    else:
        incumbent = type('OtherPlain', (scope['Plain'],), {})(incumbent.weight)
    assert scope['select_head'](incumbent) is incumbent and probes == []


@pytest.mark.parametrize('capability', [(12, 0), (12, 2), (9, 0)])
def test_unqualified_architectures_keep_original(capability):
    scope, incumbent, _calls, probes = maintained()
    scope['torch'].cuda.get_device_capability = lambda device: (probes.append(device), capability)[1]
    assert scope['select_head'](incumbent) is incumbent and len(probes) == 1

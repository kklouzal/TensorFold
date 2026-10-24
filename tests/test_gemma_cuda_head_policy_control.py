"""Head-policy metadata controls and CPU FP8 carrier integration; no device kernels."""
import __future__

import ast
import builtins
import sys
from functools import lru_cache, partial
from pathlib import Path
from types import ModuleType
from types import SimpleNamespace as NS

import pytest

from tools.gemma4_cuda_launch_qualify import _read

SOURCE = Path(__file__).resolve().parents[1] / 'src/tensorfold/families/gemma4/cuda/head.py'
B16_SOURCE = SOURCE.parents[2] / 'qwen3_5/cuda/b16.py'
CG_SOURCE = SOURCE.with_name('head_cg.py')
CG_MODULE = 'tensorfold.families.gemma4.cuda.head_cg'


@pytest.mark.torch
def test_private_fp8_head_reads_original_weight_without_calling_prepared_cg_target(monkeypatch):
    torch = pytest.importorskip('torch')
    from tensorfold.cuda import requant
    from tensorfold.families.gemma4.cuda import head_cg
    from tensorfold.families.gemma4.cuda.head import _AdaptiveRowsHead
    from tensorfold.families.qwen3_5.cuda.weights import Plain

    def no_target(*_args, **_kwargs):
        pytest.fail('a private FP8 weight copy must not execute the BF16 target head')

    monkeypatch.setattr(head_cg, 'matmul', no_target)
    monkeypatch.setattr(head_cg, '_ext', lambda: pytest.fail('no native build in this carrier control'))
    weight = torch.ones((128, 64), dtype=torch.bfloat16)
    selected = _AdaptiveRowsHead(Plain(weight), global_cache=True)
    calls, copied = [], NS(n=128)

    def copy(original, kind):
        calls.append((original, kind))
        return copied

    # Only the owned quantizer boundary is replaced; actual Head construction
    # must read the original Tensor rather than invoke the prepared target.
    monkeypatch.setattr(requant, 'requantize', copy)
    private = requant.Head(selected, 'fp8', softcap=30.)
    assert len(calls) == 1 and calls[0][0] is weight and calls[0][1] == 'fp8'
    assert selected.weight is weight and private.linear is copied and private.softcap == 30.


def maintained(monkeypatch):
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

    def multiply_cg(value, w, *, rows):
        calls.append(('cg', rows, value, w))
        return calls[-1]

    helper = ModuleType(CG_MODULE)
    helper.matmul = multiply_cg
    monkeypatch.setitem(sys.modules, CG_MODULE, helper)
    imports = []

    def observe_import(name, _globals=None, _locals=None, fromlist=(), level=0):
        if name == 'head_cg' and level == 1:
            imports.append(name)
        return builtins.__import__(name, _globals, _locals, fromlist, level)

    probes = []
    cuda = NS(get_device_capability=lambda device: (probes.append(device), (12, 1))[1])
    scope = {'partial': partial, 'Plain': Plain, 'matmul_rows': multiply, 'cg_imports': imports,
             '__package__': 'tensorfold.families.gemma4.cuda',
             '__builtins__': {**vars(builtins), '__import__': observe_import},
             'torch': NS(bfloat16='bf16', cuda=cuda)}
    tree = ast.parse(_read(SOURCE, 1 << 20))
    tree.body = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef))]
    exec(compile(tree, str(SOURCE), 'exec', flags=__future__.annotations.compiler_flag), scope)  # noqa: S102 -- maintained AST only
    weight = NS(dtype='bf16', shape=(262144, 2816), device=NS(type='cuda'),
                is_contiguous=lambda: True, bytes=262144 * 2816 * 2)
    return scope, Plain(weight), calls, probes


@pytest.mark.parametrize('rows,expected', [(0, 16), (1, 1), (2, 4), (3, 4), (4, 4),
                                          (5, 8), (8, 8), (9, 16), (17, 16), (129, 16)])
def test_prepared_cg_policy_uses_h_geometry_with_exact_singleton_and_zero_paths(rows, expected, monkeypatch):
    scope, incumbent, calls, probes = maintained(monkeypatch)
    selected = scope['select_head'](incumbent)
    assert selected.weight is incumbent.weight and selected.incumbent is incumbent
    assert selected.nbytes() == incumbent.nbytes() and len(probes) == 1
    scope['torch'].cuda.get_device_capability = lambda _device: pytest.fail('hot capability discovery')
    value = NS(shape=(rows, 2816))
    assert selected(value) == ('cg', expected, value, incumbent.weight)
    assert calls == [('cg', expected, value, incumbent.weight)]
    assert scope['select_head'](selected) is selected
    assert scope['cg_imports'] == ['head_cg']


def test_policy_prefill_keeps_the_inherited_plain_interface(monkeypatch):
    scope, incumbent, calls, _probes = maintained(monkeypatch)
    selected, value = scope['select_head'](incumbent), object()
    assert selected.prefill(value) == ('prompt', value, incumbent.weight)
    assert calls == [('prompt', value, incumbent.weight)]


def test_zero_rows_keep_exact_native_error_without_normalization(monkeypatch):
    scope, incumbent, _calls, _probes = maintained(monkeypatch)
    primary, calls = RuntimeError('native dimensions refused'), []

    def native(value, w, *, rows):
        calls.append((value, w, rows))
        raise primary

    monkeypatch.setattr(sys.modules[CG_MODULE], 'matmul', native)
    selected, value = scope['select_head'](incumbent), NS(shape=(0, 2816))
    with pytest.raises(RuntimeError) as caught:
        selected(value)
    assert caught.value is primary and calls == [(value, incumbent.weight, 16)]


@pytest.mark.parametrize('rows,expected', [(1, 'original'), (2, 4), (5, 8), (9, 16), (17, 16)])
def test_adaptive_h_reference_keeps_original_load_path(rows, expected, monkeypatch):
    scope, incumbent, calls, _probes = maintained(monkeypatch)
    reference = scope['_AdaptiveRowsHead'](incumbent)
    value = NS(shape=(rows, 2816))
    assert reference(value) == (expected, value, incumbent.weight)
    assert calls == [(expected, value, incumbent.weight)]
    assert scope['cg_imports'] == []


@pytest.mark.parametrize('cast', [False, True])
def test_cg_wrapper_uses_private_provider_and_preserves_native_input_boundary(cast):
    calls, xarg, warg, output = [], object(), object(), object()
    converted = NS(contiguous=lambda: xarg)

    def convert(dtype):
        calls.append(('cast', dtype))
        return converted

    value = NS(dtype='fp32' if cast else 'bf16', to=convert, contiguous=lambda: xarg)
    weight = NS(dtype='bf16', contiguous=lambda: warg)

    def native(x, w, rows):
        calls.append(('native', x, w, rows))
        return output

    tree = ast.parse(_read(CG_SOURCE, 1 << 20))
    tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'matmul']
    scope = {'_ext': lambda: NS(gemma_b16_head_cg=native)}
    exec(compile(tree, str(CG_SOURCE), 'exec', flags=__future__.annotations.compiler_flag), scope)  # noqa: S102 -- maintained AST only
    assert scope['matmul'](value, weight, 4) is output
    assert calls == ([('cast', 'bf16')] if cast else []) + [('native', xarg, warg, 4)]


def test_prepared_cg_callable_stays_bound_for_single_and_multirow_calls(monkeypatch):
    scope, incumbent, calls, _probes = maintained(monkeypatch)
    selected = scope['select_head'](incumbent)
    monkeypatch.setattr(sys.modules[CG_MODULE], 'matmul', lambda *_args, **_kwargs: pytest.fail('hot policy rebinding'))
    for rows in (1, 5):
        value = NS(shape=(rows, 2816))
        assert selected(value) == ('cg', 1 if rows == 1 else 8, value, incumbent.weight)
    assert len(calls) == 2
    assert scope['cg_imports'] == ['head_cg']


def test_private_recipe_versions_authority_without_changing_normal_factory(monkeypatch):
    calls = []

    def load(**options):
        module = NS(options=options)
        calls.append(module)
        return module

    build = ModuleType('tensorfold.cuda.build')
    build.load = load
    monkeypatch.setitem(sys.modules, build.__name__, build)
    factories = []
    for source in (B16_SOURCE, CG_SOURCE):
        tree = ast.parse(_read(source, 1 << 20))
        tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == '_ext']
        scope = {'lru_cache': lru_cache, 'Path': Path, '__file__': str(source), 'b16': NS(__file__=str(B16_SOURCE))}
        exec(compile(tree, str(source), 'exec'), scope)  # noqa: S102 -- actual factory body, fake build boundary
        factories.append(scope['_ext'])
    normal, cached = factories[0](), factories[1]()
    assert factories[1]() is cached and len(calls) == 2
    here = B16_SOURCE.parent
    assert normal.options == {'name': 'tensorfold_qwen_b16_v6',
        'sources': [str(here / 'b16.cpp'), str(here / 'b16.cu')],
        'extra_cuda_cflags': ['-O3'], 'verbose': False}
    assert cached.options == {'name': 'tensorfold_gemma_b16_cg_v1',
        'sources': [str(here / 'gemma_head_cg.cpp'), str(here / 'b16.cu')],
        'extra_cuda_cflags': ['-O3', '-DTENSORFOLD_GEMMA_HEAD_CG'], 'verbose': False}


@pytest.mark.parametrize('reuse_rows', [2, 4, 8, 16])
def test_fixed_comparator_policy_and_metadata_are_preserved(reuse_rows, monkeypatch):
    scope, incumbent, calls, _probes = maintained(monkeypatch)
    prepared = scope['prepare_head'](incumbent, reuse_rows=reuse_rows)
    value = NS(shape=(5, 2816))
    assert prepared.reuse_rows == reuse_rows and prepared.incumbent is incumbent
    assert prepared(value) == (reuse_rows, value, incumbent.weight)
    assert calls == [(reuse_rows, value, incumbent.weight)]
    assert scope['cg_imports'] == []


@pytest.mark.parametrize('change', ['dtype', 'shape', 'host', 'contiguous', 'subclass'])
def test_unqualified_inputs_keep_original_without_new_capability_discovery(change, monkeypatch):
    scope, incumbent, _calls, probes = maintained(monkeypatch)
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
    assert scope['cg_imports'] == []


@pytest.mark.parametrize('capability', [(12, 0), (12, 2), (9, 0)])
def test_unqualified_architectures_keep_original(capability, monkeypatch):
    scope, incumbent, _calls, probes = maintained(monkeypatch)
    scope['torch'].cuda.get_device_capability = lambda device: (probes.append(device), capability)[1]
    assert scope['select_head'](incumbent) is incumbent and len(probes) == 1
    assert scope['cg_imports'] == []

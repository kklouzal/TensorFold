"""Maintained Flash HC role/launch controls; no Torch or device execution."""

import __future__

import ast
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from tensorfold.cleanup import rollback

ROOT = Path(__file__).resolve().parents[1]
CUDA = ROOT / 'src/tensorfold/families/qwen4_exp/cuda'


def source_bytes(path):
    stream = path.open('rb')
    close = stream.close
    try:
        raw = stream.read()
    except BaseException as primary:  # noqa: BLE001 -- preserve the source-read root through adopted rollback
        try:
            rollback(stream, primary, close)
        except BaseException as failure:
            if failure is primary:
                raise
            raise primary from failure
    else:
        close()
        return raw


def definitions(filename, names, namespace):
    tree = ast.parse(source_bytes(CUDA / filename))
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in nodes} == set(names)
    exec(compile(ast.Module(nodes, []), str(CUDA / filename), 'exec',  # noqa: S102 -- maintained source only
                 flags=__future__.annotations.compiler_flag), namespace)
    return namespace


def prepared_policy(quant, prefill, exl3=False):
    path = CUDA / 'state.py'
    tree = ast.parse(source_bytes(path))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'Buffers')
    initialize = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == '__init__')
    first_tensor = next(index for index, node in enumerate(initialize.body) if isinstance(node, ast.Assign)
                        and any(isinstance(target, ast.Attribute) and target.attr == 'ids' for target in node.targets))
    initialize.body = initialize.body[:first_tensor]
    scope = {'ENDS': 16, 'torch': NS(bfloat16='bf16', float32='f32')}
    exec(compile(ast.Module([initialize], []), str(path), 'exec',  # noqa: S102 -- maintained source only
                 flags=__future__.annotations.compiler_flag), scope)
    buffers = NS()
    weights = NS(cfg=NS(hidden=2560, streams=4, quant=quant), device='cuda:0', x3=object() if exl3 else None)
    scope['__init__'](buffers, weights, 8, 64, prefill=prefill)
    return buffers.mlp_mix_sums


def test_source_read_failure_retains_exact_primary_and_close_root():
    primary, cleanup = OSError('source read failed'), OSError('source close failed')
    calls = []

    def read():
        calls.append('read')
        raise primary

    def close():
        calls.append('close')
        raise cleanup

    stream = NS(read=read, close=close)
    with pytest.raises(OSError) as caught:
        source_bytes(NS(open=lambda _mode: stream))
    assert caught.value is primary and primary.__cause__ is cleanup
    assert calls == ['read', 'close']


@pytest.mark.parametrize('quant', ['modelopt', 'mlx', 'exl3'])
@pytest.mark.parametrize('prefill', [False, True])
@pytest.mark.parametrize('exl3', [False, True])
def test_prepared_policy_omits_only_modelopt_decode_without_exl3(quant, prefill, exl3):
    expected = not (quant == 'modelopt' and not prefill and not exl3)
    assert prepared_policy(quant, prefill, exl3) is expected


class Rows:
    def __init__(self, shape=(2, 10240)):
        self.shape, self.device = shape, 'cuda:0'

    def __getitem__(self, _key):
        return self


@pytest.mark.parametrize('sums', [False, True])
@pytest.mark.parametrize('linear', [False, True])
@pytest.mark.parametrize('pending_mode', [None, 1, 2])
@pytest.mark.parametrize('mtp', [False, True])
def test_pre_moe_passes_role_only_to_mlp_readout(sums, linear, pending_mode, mtp):
    calls = []
    namespace = definitions('forward.py', ['_pre_moe'], {
        'hc_block': lambda *args, **options: calls.append((args, options)),
        'gdn_block': lambda *_args: (1, 'gdn-branch'),
        'attn_block': lambda *_args: (1, 'attention-branch'),
    })
    rows = Rows()
    b = NS(h=rows, inj_a=rows, inj_m=rows, mlp_mix_sums=sums)
    layer = NS(ple=None, linear=linear, attn_hc='attention-HC', mlp_hc='MLP-HC')
    w = NS(cfg=NS(eps=1e-6, streams=4, low=320))
    pending = None if pending_mode is None else (pending_mode, 'previous', 'weights', rows)
    namespace['_pre_moe'](layer, w, (), b, 2, pending, mtp=mtp)
    assert len(calls) == 2
    assert calls[0][0][0] == 'attention-HC' and 'mixed_sums' not in calls[0][1]
    assert calls[1][0][0] == 'MLP-HC' and calls[1][1]['mixed_sums'] is sums
    assert calls[1][1]['branch'] == ('gdn-branch' if linear else 'attention-branch')


@pytest.mark.parametrize('sums', [None, False, True])
def test_b16_readout_preserves_all_phases_and_default_sums(sums):
    calls = []
    rows = Rows()
    mix_flags = []
    glue = NS(hc_writeback=lambda *_a, **_k: calls.append('writeback'),
              hc_normed=lambda *_a: calls.append('norm'), hc_act=lambda *_a: calls.append('act'),
              hc_mix=lambda *_a, **k: (calls.append('mix'), mix_flags.append(k['sums'])))

    def down(*_args, **_kwargs):
        calls.append('down')
        return rows

    namespace = definitions('forward.py', ['hc_block', '_readout', '_readout_b16'], {
        'glue': glue, 'FUSED_ROWS': 16, 'qmm': NS(Q4=type('Q4', (), {})),
        'bf16': NS(matmul=down), 'torch': NS(empty=lambda *_a, **_k: rows, float32='f32'),
        '_mm': lambda *_a: calls.append('up'),
    })
    b = NS(prefill=False, pss=rows, normed=rows, xs_normed=rows, act=rows, xs_act=rows,
           up=rows, mixed=rows, xs_mixed=rows)
    hc = NS(inject=True, scale=object(), down=NS(kernel='b16', b=object(), n=324), up=object())
    options = {} if sums is None else {'mixed_sums': sums}
    namespace['hc_block'](hc, b, 2, 1e-6, 4, 320, 1, None, rows, rows, branch=rows, **options)
    assert calls == ['writeback', 'norm', 'down', 'act', 'up', 'mix']
    assert mix_flags == [True if sums is None else sums]


def test_non_b16_readout_keeps_original_plain_call_interface():
    calls = []
    namespace = definitions('forward.py', ['_readout'], {
        '_readout_plain': lambda *args, **options: calls.append((args, options)),
    })
    args = (NS(down=NS(kernel='qmm')), object(), object(), 2, 1e-6, 4, 320, None)
    namespace['_readout'](*args)
    assert calls == [(args, {})]


@pytest.mark.parametrize('sums', [None, False, True])
def test_mix_wrapper_selects_constexpr_without_changing_launch(sums):
    calls = []

    class Kernel:
        def __getitem__(self, grid):
            return lambda *args, **options: calls.append((grid, args, options))

    namespace = definitions('glue.py', ['hc_mix'], {'_hc_mix': Kernel(), 'pdl': NS(ON=True)})
    inputs = (Rows((3, 10240)), object(), object(), object(), 4)
    namespace['hc_mix'](*inputs, **({} if sums is None else {'sums': sums}))
    grid, args, options = calls[0]
    assert grid == (3, 10) and args == inputs[:4]
    assert options == {'D': 2560, 'S': 4, 'BLOCK': 256, 'SUMS': True if sums is None else sums,
                       'num_warps': 2, 'PDL': True, 'launch_pdl': True}

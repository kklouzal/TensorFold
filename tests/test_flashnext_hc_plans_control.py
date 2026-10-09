"""Maintained HC control/hook/lifetime contracts with stdlib-only SDK doubles."""
import ast
from dataclasses import dataclass
import gc
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import patch
import weakref


ROOT = Path(__file__).resolve().parents[1]


def maintained():
    path = ROOT/'src/tensorfold/families/qwen4_exp/cuda/hc_plans.py'
    nodes = [node for node in ast.parse(path.read_bytes()).body if isinstance(node, ast.ClassDef)]
    namespace = {'dataclass': dataclass}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['_F16Plan'], namespace['_HCPlans']


class JIT:
    def __init__(self):
        self.pre_run_hooks = []
        self.calls = []

    def __getitem__(self, grid):
        def run(*args, **kwargs):
            self.calls.append((grid, args, kwargs))
        return run


def test_f16_dynamic_pre_run_hook_retains_original_argument_packing():
    fixed, _ = maintained()
    jit, calls = JIT(), []
    kernel = SimpleNamespace(src=SimpleNamespace(fn=jit))
    arguments = tuple(range(12))+(True,)
    plan = fixed(kernel, lambda *args: calls.append(args), arguments, (), {'grid': (1, 2, 3)})
    plan()
    assert calls == [arguments] and not jit.calls
    jit.pre_run_hooks.append(object())
    plan()
    grid, positional, keywords = jit.calls[0]
    assert grid == (1, 2, 3) and positional == arguments[:7]
    assert keywords == dict(K=7, KS=8, BM=9, BN=10, BK=11, F32=True, num_warps=4, num_stages=3)


def test_hook_registered_during_down_is_honored_by_up():
    fixed, _ = maintained()
    jit, compiled = JIT(), []
    kernel = SimpleNamespace(src=SimpleNamespace(fn=jit))
    arguments = tuple(range(12))+(False,)
    def down(*args):
        compiled.append('down')
        jit.pre_run_hooks.append(object())
    a = fixed(kernel, down, arguments, (), {'grid': (1, 1, 1)})
    b = fixed(kernel, lambda *args: compiled.append('up'), arguments, (), {'grid': (1, 1, 1)})
    a()
    b()
    assert compiled == ['down'] and len(jit.calls) == 1


def test_exact_readout_phase_and_normed_skip():
    _, manager = maintained()
    plans = manager.__new__(manager)
    calls = []
    glue = SimpleNamespace(hc_normed=lambda *a: calls.append('norm'),
                           hc_reduce_act=lambda *a: calls.append('ordered-reduce-act'),
                           hc_mix=lambda *a: calls.append('mix'))
    hc = SimpleNamespace(scale=object())
    entry = (lambda: calls.append('down'), lambda: calls.append('up'), *[object() for _ in range(9)])
    plans._entries = {id(hc): (None, entry)}
    plans._glue, plans._jit, plans._closed = glue, JIT(), False
    assert plans.enabled
    plans(hc, [1], 1, .01, 4, 320, None)
    assert calls == ['norm', 'down', 'ordered-reduce-act', 'up', 'mix']
    calls.clear()
    plans(hc, [1], 1, .01, 4, 320, None, normed=True)
    assert calls == ['down', 'ordered-reduce-act', 'up', 'mix']
    plans._jit.pre_run_hooks.append(object())
    assert not plans.enabled


class Owner:
    pass


def test_close_failed_fence_retains_owners_then_retry_releases():
    _, manager = maintained()
    plans = manager.__new__(manager)
    owner = Owner()
    ref = weakref.ref(owner)
    plans._entries, plans._owners = {'view': owner}, (owner,)
    plans._device, plans._closed = 'owned-device', False
    del owner
    calls = []
    def sync(device):
        calls.append(device)
        if len(calls) == 1:
            raise RuntimeError('controlled fence failure')
    torch = ModuleType('torch')
    torch.cuda = SimpleNamespace(synchronize=sync)
    with patch.dict(sys.modules, {'torch': torch}):
        try:
            plans.close()
        except RuntimeError:
            assert not plans._closed and plans._entries and plans._owners
        else:
            raise AssertionError('failed fence released owners')
        gc.collect()
        assert ref() is not None
        plans.close()
        plans.close()
    gc.collect()
    assert ref() is None and plans._closed and not plans._entries and plans._owners == ()
    assert calls == ['owned-device', 'owned-device']


def test_forward_uses_owned_plan_only_when_enabled():
    path = ROOT/'src/tensorfold/families/qwen4_exp/cuda/forward.py'
    fn = next(n for n in ast.parse(path.read_bytes()).body if isinstance(n, ast.FunctionDef)
              and n.name == '_readout_plain')
    calls = []
    namespace = {'glue': SimpleNamespace(hc_normed=lambda *a: calls.append('original-norm'),
                                       hc_mix=lambda *a: calls.append('original-mix')),
                 '_down_act': lambda *a: calls.append('original-down'),
                 '_mm': lambda *a: calls.append('original-up')}
    source = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), fn],
                        type_ignores=[])
    ast.fix_missing_locations(source)
    exec(compile(source, str(path), 'exec'), namespace)
    class Plans:
        enabled = True
        def __call__(self, *args, **kwargs):
            calls.append('owned-plan')
    plans = Plans()
    b = SimpleNamespace(hc_plans=plans, prefill=False,
                        **{key: [object()] for key in ('pss', 'normed', 'xs_normed', 'act', 'xs_act',
                                                     'up', 'mixed', 'xs_mixed')})
    hc = SimpleNamespace(scale=object(), up=object())
    namespace['_readout_plain'](hc, b, [1], 1, .01, 4, 320, None)
    assert calls == ['owned-plan']
    plans.enabled = False
    calls.clear()
    namespace['_readout_plain'](hc, b, [1], 1, .01, 4, 320, None)
    assert calls == ['original-norm', 'original-down', 'original-up', 'original-mix']

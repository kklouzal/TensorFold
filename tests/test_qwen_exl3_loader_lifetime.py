"""Actual EXL3 loader reader transport under explicit opaque stdlib providers.

No tensor runtime/math result or real checkpoint/model validity is claimed.
"""
import ast
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from test_qwen_loader_lifetime import Value

REPO = Path(__file__).resolve().parents[1]
BASE = REPO / 'src/tensorfold/families/qwen3_5/cuda'


class Ownership(unittest.TestCase):
    def execute(self, primary=None, cleanup=None, *, groups=False, erase=False, allocation=None):
        calls = []
        class Reader:
            def close(self):
                calls.append('close')
                if erase:
                    primary.__cause__ = primary.__context__ = None
                if cleanup is not None:
                    raise cleanup
        reader = Reader()
        cfg = SimpleNamespace(layers=0, rope_dims=4, rope_theta=10000)
        def plain(*args, **kwargs):
            calls.append('construct')
            return SimpleNamespace()
        weights = ModuleType('tensorfold.families.qwen3_5.cuda.weights')
        weights.__dict__.update(Config=SimpleNamespace(read=lambda _: cfg), GDN=plain, Attention=plain,
                               Layer=plain, Plain=plain, Weights=lambda **kw: SimpleNamespace(layers=[]))
        ckpt = SimpleNamespace(bad={}, groups={'lm_head': object()}, plain=[])
        fmt = ModuleType('tensorfold.cuda.exl3.format')
        fmt.scan = lambda *args, **kwargs: ckpt
        package = ModuleType('tensorfold.cuda.exl3')
        package.format = fmt
        prefill = ModuleType('tensorfold.cuda.exl3.prefill')
        prefill.Workspace = lambda: object()
        modules = {m.__name__: m for m in (weights, fmt, package, prefill)}
        def read(*args):
            calls.append('read')
            if primary is not None and not groups:
                raise primary
            return {'model.language_model.embed_tokens.weight': Value(),
                    'model.language_model.norm.weight': Value()}
        def read_groups(*args):
            calls.append('groups')
            if primary is not None and groups:
                raise primary
            return {'lm_head': object()}
        helper_tree = ast.parse((BASE / 'weights.py').read_text())
        helpers = [n for n in helper_tree.body if isinstance(n, ast.FunctionDef)
                   and n.name in ('_close_failed_checkpoint', '_close_failed_checkpoint_impl')]
        scope = {'__name__': 'tensorfold.families.qwen3_5.cuda.opaque',
                 '__package__': 'tensorfold.families.qwen3_5.cuda', 'Path': Path,
                 '_where': lambda _: {}, '_files': lambda *args: reader,
                 '_read': read, '_read_groups': read_groups,
                 'torch': SimpleNamespace(float64='f64', float32='f32', arange=lambda *a, **kw: Value())}
        if allocation is not None:
            def fail(*args):
                raise allocation
            scope['BaseExceptionGroup'] = fail
        load = next(n for n in ast.parse((BASE / 'exl3_load.py').read_text()).body
                    if isinstance(n, ast.FunctionDef) and n.name == 'load_exl3')
        exec(compile(ast.Module(body=[*helpers, load], type_ignores=[]), '<actual-exl3-owner>', 'exec',
                     flags=__import__('__future__').annotations.compiler_flag), scope)
        weights._close_failed_checkpoint = scope['_close_failed_checkpoint']
        with patch.dict(sys.modules, modules):
            try:
                result = scope['load_exl3'](Path('/opaque-checkpoint'), device='cpu')
            except BaseException as actual:
                return actual, calls
        return result, calls

    def test_actual_plain_and_group_read_failure_drains_once_preserving_primary(self):
        for groups in (False, True):
            with self.subTest(groups=groups):
                primary = KeyboardInterrupt()
                actual, calls = self.execute(primary, groups=groups)
                self.assertIs(actual, primary)
                self.assertEqual(calls.count('close'), 1)
                self.assertNotIn('construct', calls)

    def test_malformed_notes_and_close_failure_keep_primary_native_and_every_status(self):
        primary, native, context, cleanup = KeyboardInterrupt(), LookupError(), EOFError(), OSError()
        primary.__cause__, primary.__context__, primary.__notes__ = native, context, 123
        actual, calls = self.execute(primary, cleanup)
        self.assertIs(actual, primary)
        self.assertEqual(actual.__cause__.exceptions[:3], (native, context, cleanup))
        self.assertIsInstance(actual.__cause__.exceptions[-1], TypeError)
        self.assertEqual(calls.count('close'), 1)

    def test_same_primary_erasing_close_keeps_captured_native_context_once(self):
        primary, native, context = KeyboardInterrupt(), LookupError(), EOFError()
        primary.__cause__, primary.__context__ = native, context
        actual, calls = self.execute(primary, primary, erase=True)
        self.assertIs(actual, primary)
        self.assertEqual(actual.__cause__.exceptions, (native, context))
        self.assertEqual(calls.count('close'), 1)

    def test_actual_group_exhaustion_retains_primary_and_source_status_frames(self):
        primary, native, context, cleanup = KeyboardInterrupt(), LookupError(), EOFError(), OSError()
        allocation = MemoryError('owned grouping failure')
        primary.__cause__, primary.__context__ = native, context
        actual, calls = self.execute(primary, cleanup, erase=True, allocation=allocation)
        self.assertIs(actual, primary)
        self.assertIs(actual.__cause__, allocation)
        frames, trace = [], actual.__traceback__
        while trace is not None:
            frames.append(trace.tb_frame)
            trace = trace.tb_next
        outer = next(f.f_locals for f in frames if f.f_code.co_name == '_close_failed_checkpoint')
        self.assertIs(outer['native_cause'], native)
        self.assertIs(outer['native_context'], context)
        trace = allocation.__traceback__
        while trace is not None and trace.tb_frame.f_code.co_name != '_close_failed_checkpoint_impl':
            trace = trace.tb_next
        self.assertIsNotNone(trace)
        self.assertEqual(trace.tb_frame.f_locals['others'], [native, context, cleanup])
        self.assertEqual(calls.count('close'), 1)

    def test_success_closes_before_construction_and_normal_close_fault_fails_operation(self):
        result, calls = self.execute()
        self.assertTrue(hasattr(result, 'inv_freq'))
        self.assertEqual(calls.count('close'), 1)
        self.assertLess(calls.index('close'), calls.index('construct'))
        cleanup = OSError()
        actual, calls = self.execute(cleanup=cleanup)
        self.assertIs(actual, cleanup)
        self.assertEqual(calls.count('close'), 1)
        self.assertNotIn('construct', calls)


if __name__ == '__main__':
    unittest.main()

"""Bonsai current-value/operation lifetime controls; no MLX or tensor math."""
import ast
import gc
from pathlib import Path
from types import SimpleNamespace
import unittest
import weakref

from tensorfold.kernels.qwen.dense.v1 import projection_operation as operation

ROOT = Path(__file__).resolve().parents[1]
TREE = ast.parse((ROOT / 'src/tensorfold/families/bonsai/modules.py').read_bytes())
CACHE = next(node for node in TREE.body if isinstance(node, ast.ClassDef) and node.name == 'RotationCache')


class Array:
    """A descriptor/value token; mutation models the documented MLX identity case."""
    def __init__(self, value):
        self.value = value


def fixture():
    calls = []
    def rotate(x, signs):
        calls.append((x.value, signs.value))
        return Array((x.value, signs.value))
    ns = {'rotate': SimpleNamespace(rotate_rows=rotate)}
    owned = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), CACHE],
                       type_ignores=[])
    exec(compile(ast.fix_missing_locations(owned), '<actual-bonsai-RotationCache>', 'exec'), ns)
    return ns['RotationCache'](), calls


class Authority(unittest.TestCase):
    def test_generic_calls_observe_same_object_input_and_signs_descriptor_changes(self):
        cache, calls = fixture()
        x, signs = Array('x1'), Array('s1')
        first = cache(x, signs)
        x.value = 'x2'
        second = cache(x, signs)
        signs.value = 's2'
        third = cache(x, signs)
        self.assertEqual([first.value, second.value, third.value], [('x1', 's1'), ('x2', 's1'), ('x2', 's2')])
        self.assertEqual(len(calls), 3)
        self.assertFalse(hasattr(cache, 'last'))

    def test_siblings_reuse_only_one_owned_operation_and_next_operation_is_fresh(self):
        cache, calls = fixture()
        x, signs = Array('x1'), Array('s1')
        with operation.operation():
            first = cache(x, signs)
            with operation.operation():
                self.assertIs(cache(x, signs), first)
            self.assertIs(cache(x, signs), first)
        x.value = 'x2'
        with operation.operation():
            self.assertEqual(cache(x, signs).value, ('x2', 's1'))
        self.assertEqual(len(calls), 2)

    def test_exact_transform_owner_and_current_input_signs_identity(self):
        first, a = fixture()
        second, b = fixture()
        x, signs = Array('x'), Array('s')
        with operation.operation():
            y = first(x, signs)
            self.assertIsNot(first(Array('x'), signs), y)
            self.assertIsNot(first(x, Array('s')), y)
            self.assertIsNot(second(x, signs), y)
        self.assertEqual((len(a), len(b)), (3, 1))

    def test_success_and_interrupt_scope_retire_all_borrowed_arrays(self):
        for interrupted in (False, True):
            with self.subTest(interrupted=interrupted):
                cache, calls = fixture()
                refs = []
                try:
                    with operation.operation():
                        x, signs = Array('x'), Array('s')
                        y = cache(x, signs)
                        refs.extend((weakref.ref(x), weakref.ref(signs), weakref.ref(y)))
                        del x, signs, y
                        self.assertTrue(all(ref() is not None for ref in refs))
                        if interrupted:
                            raise KeyboardInterrupt('owned scope interrupted')
                except KeyboardInterrupt:
                    if not interrupted:
                        raise
                gc.collect()
                self.assertTrue(all(ref() is None for ref in refs))


if __name__ == '__main__':
    unittest.main()

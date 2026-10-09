"""Execute actual borrowed conv/SSM pair selection with opaque metadata."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
TREE = ast.parse((ROOT / 'src/tensorfold/kernels/nemotron/lightning/v1/kernels.py').read_bytes())
OWNER = next(node for node in TREE.body if isinstance(node, ast.ClassDef) and node.name == 'FusedDecode')
METHOD = next(node for node in OWNER.body if isinstance(node, ast.FunctionDef) and node.name == '_states_in')


def actual(caches):
    calls = []
    def concatenate(parts):
        calls.append(tuple(parts))
        return tuple(parts)
    namespace = {'mx': SimpleNamespace(concatenate=concatenate)}
    future = ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[future, METHOD], type_ignores=[])),
                 '<actual-Nemotron-pair-owner>', 'exec'), namespace)
    owner = SimpleNamespace(_mamba_states=lambda cache, dtype: (cache.conv, cache.ssm))
    return namespace['_states_in'](owner, caches, 'opaque_dtype'), calls


class PairAuthority(unittest.TestCase):
    def test_common_ssm_with_distinct_conv_uses_each_current_conv_state(self):
        conv0, conv1, ssm = object(), object(), object()
        caches = [SimpleNamespace(ref=(conv0, ssm, 1), conv='conv0-row1', ssm='ssm-row1'),
                  SimpleNamespace(ref=(conv1, ssm, 2), conv='conv1-row2', ssm='ssm-row2')]
        result, calls = actual(caches)
        self.assertEqual(result, (('conv0-row1', 'conv1-row2'), ('ssm-row1', 'ssm-row2'), None))
        self.assertEqual(len(calls), 2)

    def test_shared_complete_pair_borrows_once_with_original_slot_order(self):
        conv, ssm = object(), object()
        caches = [SimpleNamespace(ref=(conv, ssm, row)) for row in (2, 0, 2, 1)]
        result, calls = actual(caches)
        self.assertIs(result[0], conv)
        self.assertIs(result[1], ssm)
        self.assertEqual(result[2], (2, 0, 2, 1))
        self.assertEqual(calls, [])

    def test_differing_ssm_or_absent_reference_takes_current_owned_stack(self):
        for kind in ('ssm', 'missing'):
            with self.subTest(kind=kind):
                conv, ssm = object(), object()
                refs = [(conv, ssm, 0), (conv, object(), 1) if kind == 'ssm' else None]
                caches = [SimpleNamespace(ref=ref, conv=i, ssm=i + 10) for i, ref in enumerate(refs)]
                result, calls = actual(caches)
                self.assertEqual(result, ((0, 1), (10, 11), None))
                self.assertEqual(len(calls), 2)


if __name__ == '__main__':
    unittest.main()

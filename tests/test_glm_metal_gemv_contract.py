"""Owned GLM GEMV admission, cohort and writer contracts without MLX.

Integer oracles prove work partition/addresses and source publication ordering;
they do not compile Metal or establish native numeric or model equivalence.
"""
import ast
from collections import Counter
from operator import index
from pathlib import Path
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'src/tensorfold/kernels/glm/flash/v1/kernels.py'
TREE = ast.parse(SOURCE.read_bytes())
FUNCTIONS = [node for node in TREE.body if isinstance(node, ast.FunctionDef)
             and node.name in ('gemv_params', '_gemv_geometry', 'matmul_rows')]
SHADERS = {node.targets[0].id: ast.literal_eval(node.value) for node in TREE.body
           if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
           and node.targets[0].id in ('_GEMV_ROWS', '_GEMV_T_ROWS')}
NS = {'index': index, 'mx': SimpleNamespace(array=object)}
exec(compile(ast.Module(body=FUNCTIONS, type_ignores=[]), '<actual-glm-gemv-host>', 'exec'), NS)


def output_owners(transposed, n, params):
    """Independent lexical output-tile oracle, including every physical lane.

    Retained original loads may overlap; only the original logical tile owns
    writes. Uniform multi-simdgroup publication requires every physical lane.
    """
    bm, bn, sm, sn, tm, tn = params
    width = bn * sn * tn if transposed else bm * sm * tm
    owned, cohorts = [], []
    for group in range(-(-n // width)):
        arrivals = 0
        for simd in range(bm * bn):
            for lane in range(32):
                thr_m, thr_n = lane // sn, lane % sn
                sg_m, sg_n = simd // bn, simd % bn
                original = group * width + ((sn * sg_n + thr_n) * tn if transposed else (sm * sg_m + thr_m) * tm)
                tile = tn if transposed else tm
                live = original < n
                if transposed:
                    arrivals += 1  # no early return, independent of input loop
                    if not live:
                        continue
                    start = original if original + tile < n else n - tile
                    if sg_m or thr_m:
                        continue
                else:
                    if live or bn > 1:
                        arrivals += 1
                    start = original if original + tile <= n else n - tile
                    if not live or sg_n or thr_n:
                        continue
                assert 0 <= start <= n - tile
                owned.extend(at for at in range(start, start + tile) if original <= at < n)
        if bm > 1 and transposed or bn > 1 and not transposed:
            cohorts.append(arrivals)
    return Counter(owned), cohorts


class Geometry(unittest.TestCase):
    def test_explicit_empty_params_refused_before_cast_or_kernel(self):
        class Tensor:
            ndim, dtype, shape = 2, 'BF16', (1, 4)
            def astype(self, dtype):
                raise AssertionError('empty params reached a cast')
        NS['metal'] = lambda: True
        NS['mx'] = SimpleNamespace(float32='FP32', bfloat16='BF16')
        x, m = Tensor(), Tensor()
        m.shape = (5, 4)
        with self.assertRaises(ValueError):
            NS['matmul_rows'](x, m, transposed=False, params=())

    def test_every_logical_output_one_writer_and_all_publication_cohorts_arrive(self):
        count = 0
        for transposed in (False, True):
            for k in (0, 1, 63, 64, 65, 2048, 8192):
                for n in (1, 2, 3, 4, 5, 7, 15, 16, 17, 63, 64, 65, 127, 128, 129, 511, 512, 513, 2048, 4096):
                    params = NS['gemv_params'](transposed, k, n)
                    NS['_gemv_geometry'](3, k, n, transposed, params)
                    owners, arrivals = output_owners(transposed, n, params)
                    self.assertEqual(owners, Counter({i: 1 for i in range(n)}))
                    self.assertTrue(all(value == 32 * params[0] * params[1] for value in arrivals))
                    count += 1
        for params in ((2, 2, 4, 8, 2, 3), (4, 2, 8, 4, 1, 2), (1, 8, 1, 32, 3, 2)):
            for transposed in (False, True):
                for n in (9, 17, 65, 131):
                    NS['_gemv_geometry'](1, 129, n, transposed, params)
                    owners, arrivals = output_owners(transposed, n, params)
                    self.assertEqual(owners, Counter({i: 1 for i in range(n)}))
                    self.assertTrue(all(value == 32 * params[0] * params[1] for value in arrivals))
                    count += 1
        self.assertEqual(count, 304)

    def test_invalid_static_metadata_refused_before_cast_or_shader_construction(self):
        limits = [((1, 1, 1, 32, 4, 4), 3, 5, 2), ((1, 1, 1, 31, 1, 1), 1, 1, 1),
                  ((33, 1, 1, 32, 1, 1), 1, 1, 100), ((1, 32, 8, 4, 64, 8), 1, 64, 8192),
                  ((1, 1, 1, 32, 1, 2**31), 1, 1, 2**31 - 1),
                  ((1, 1, 1, 32, 1, 3), 1, 2**31 - 1, 129),
                  ((1, 1, 1, 32, 4, 1), 1, 2**30, 129),
                  ((1, 1, 1, 32, 1, 1), 1, 1, 2**31 - 1),
                  ((True, 1, 1, 32, 1, 1), 1, 1, 1), ((1., 1, 1, 32, 1, 1), 1, 1, 1),
                  ((1, 1, 1, 32, 1, 1), -1, 1, 1), ((1, 1, 1, 32, 1, 1), 1, -1, 1),
                  ((1, 1, 1, 32, 1, 1), 1, 1, 2**31)]
        for params, rows, k, n in limits:
            with self.subTest(params=params, rows=rows, k=k, n=n), self.assertRaises(ValueError):
                NS['_gemv_geometry'](rows, k, n, False, params)
        class Tensor:
            ndim, dtype, shape = 2, 'BF16', (1, 4)
            def astype(self, dtype):
                self.fail('invalid geometry reached a cast')
        NS['metal'] = lambda: True
        NS['mx'] = SimpleNamespace(float32='FP32', bfloat16='BF16')
        x, m = Tensor(), Tensor()
        m.shape = (5, 4)
        with self.assertRaises(ValueError):
            NS['matmul_rows'](x, m, transposed=False, params=(1, 1, 1, 31, 1, 1))

    def test_defaults_custom_index_counts_and_empty_geometry_keep_supported_domain(self):
        class Count:
            def __index__(self):
                return 1
        self.assertEqual(NS['_gemv_geometry'](0, 0, 0, True, (Count(), 1, 1, 32, 1, 1)), (1, 1, 1, 32, 1, 1))
        for transposed in (False, True):
            for k, n in ((0, 0), (1, 0), (0, 1), (2**31 - 129, 128)):
                params = NS['gemv_params'](transposed, k, n)
                NS['_gemv_geometry'](0, k, n, transposed, params)


class SourceOrdering(unittest.TestCase):
    def test_both_shared_publication_barriers_are_uniform_memory_fences(self):
        for transposed, name in ((False, '_GEMV_ROWS'), (True, '_GEMV_T_ROWS')):
            shader = SHADERS[name]
            self.assertNotIn('mem_flags::mem_none', shader)
            self.assertEqual(shader.count('threadgroup_barrier(mem_flags::mem_threadgroup);'), 1)
            before, after = shader.split('threadgroup_barrier(mem_flags::mem_threadgroup);')
            self.assertIn('tgp_results[', before)
            self.assertIn('}\n    ', before[-12:])
            self.assertIn('result[', after)
            if transposed:
                self.assertNotIn('return;', shader)
                self.assertIn('if (BM > 1)', before)
                self.assertIn('if (thrM == 0 && sgM == 0)', after)
            else:
                self.assertIn('if (!live && BN == 1) return;', shader)
                self.assertIn('if (BN > 1)', before)
                self.assertIn('if (thrN == 0 && sgN == 0)', after)

    def test_tail_stores_require_original_lexical_tile_ownership(self):
        self.assertIn('const int write_col = out_col;', SHADERS['_GEMV_T_ROWS'])
        self.assertIn('out_col + j >= write_col && out_col + j < out_vec_size', SHADERS['_GEMV_T_ROWS'])
        self.assertIn('const int write_row = out_row;', SHADERS['_GEMV_ROWS'])
        self.assertIn('out_row + tm >= write_row && out_row + tm < out_vec_size', SHADERS['_GEMV_ROWS'])


if __name__ == '__main__':
    unittest.main()

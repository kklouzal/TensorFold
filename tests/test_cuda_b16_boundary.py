"""B16 host range/configuration controls using stdlib only."""
from __future__ import annotations

import itertools
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'src/tensorfold/families/qwen3_5/cuda/b16.cu'
MAXIMUM = 2**31 - 1


class Boundary(unittest.TestCase):
    def test_actual_dimension_predicate_matches_independent_integer_oracle(self):
        text = SOURCE.read_text()
        predicate = re.search(r'TORCH_CHECK\((x.size\(0\) >= 1.*?),\s*"B16 dimensions', text, re.S).group(1)
        predicate = predicate.replace('x.size(0)', 'm').replace('x.size(1)', 'k').replace('w.size(0)', 'n')
        predicate = predicate.replace('(prompt ? 0 : 256)', '(0 if prompt else 256)')
        predicate = predicate.replace('&&', ' and ')
        code = compile(' '.join(predicate.split()), 'actual-B16-dimensions', 'eval')
        values = (0, 1, 64, MAXIMUM - 256, MAXIMUM - 127, MAXIMUM - 63, MAXIMUM, MAXIMUM + 1)
        for m, n, k, prompt in itertools.product(values, values, values, (False, True)):
            actual = bool(eval(code, {'m': m, 'n': n, 'k': k, 'prompt': prompt, 'maximum': MAXIMUM}))
            expected = 0 < m <= MAXIMUM - 127 and 0 <= n <= MAXIMUM - 63 \
                and k <= MAXIMUM - (0 if prompt else 256)
            self.assertEqual(actual, expected)

    def test_linear_loop_exit_and_padding_never_overflow_in_accepted_region(self):
        for k in (0, 8, 64, 256, (MAXIMUM - 256) // 8 * 8):
            for lane in range(32):
                first = lane * 8
                if first + 8 <= k:
                    last = first + (k - 8 - first) // 256 * 256
                    self.assertLessEqual(last + 256 + 8, MAXIMUM)
                else:
                    self.assertLessEqual(first + 8, MAXIMUM)
        for m in (1, 512, MAXIMUM - 127):
            for tile in (32, 64, 128):
                self.assertLessEqual(m + tile - 1, MAXIMUM)
        self.assertLessEqual(MAXIMUM - 63 + 64 - 1, MAXIMUM)

    def test_all_native_boundaries_validate_before_narrowing_and_allocation(self):
        text = SOURCE.read_text()
        for name in ('b16_linear', 'b16_linear_pair', 'b16_prompt', 'b16_prompt_pair'):
            marker = ('std::vector<at::Tensor> ' if name.endswith('_pair') else 'at::Tensor ') + name + '('
            body = text[text.index(marker):]
            body = body[:body.index('\n}')]
            self.assertLess(body.index('dimensions('), body.index('const int M'))
            grid = 'grid_dimensions(' if 'prompt' in name else 'linear_grid('
            self.assertLess(body.index(grid), body.index('at::empty('))
        self.assertIn('static int prompt_bm(int64_t bm, int M, int64_t N)', text)
        self.assertIn('prompt_bm(bm, M, 2 * wide)', text)
        self.assertIn('!prompt || w.size(0) > 0 || x.size(1) == 0', text)

    def test_checked_runtime_device_flags_replace_unsynchronized_boolean(self):
        text = SOURCE.read_text()
        self.assertNotIn('static bool configured', text)
        self.assertIn('configured.configure(b16_prompt_kernel<BM>, PST * pstage<BM>(), x.get_device());', text)
        header = (ROOT / 'src/tensorfold/cuda/kernels/kernel_configuration.cuh').read_text()
        self.assertIn('devices_(c10::cuda::device_count())', header)
        self.assertIn('std::call_once(flags_[device]', header)
        self.assertIn('C10_CUDA_CHECK(cudaFuncSetAttribute(', header)


if __name__ == '__main__':
    unittest.main()

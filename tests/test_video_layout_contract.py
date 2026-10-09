"""Current video layout/FP32-order controls with an independent coordinate oracle.

Only standard-library scalar/provider seams execute. Exact actual NumPy/tower
proof is retained in task artifacts; final installed image checks remain separate.
"""
from __future__ import annotations

import ast
import itertools
import math
from pathlib import Path
import struct
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]


def f32(value):
    return struct.unpack('f', struct.pack('f', value))[0]


def coordinates(shape):
    return itertools.product(*(range(n) for n in shape))


class Array:
    def __init__(self, shape, data, dtype='uint8', owned=False):
        self.shape, self.data, self.dtype, self.owned = tuple(shape), list(data), dtype, owned
        self.ndim, self.size = len(self.shape), len(self.data)
        assert math.prod(self.shape) == self.size

    def offset(self, coordinate):
        index = 0
        for n, size in zip(coordinate, self.shape):
            index = index * size + n
        return index

    def reshape(self, *shape):
        shape = list(shape)
        if -1 in shape:
            shape[shape.index(-1)] = self.size // math.prod(n for n in shape if n != -1)
        return Array(shape, self.data, self.dtype, self.owned)

    def transpose(self, *axes):
        shape = tuple(self.shape[a] for a in axes)
        values = []
        for coordinate in coordinates(shape):
            original = [0] * self.ndim
            for a, n in zip(axes, coordinate):
                original[a] = n
            values.append(self.data[self.offset(original)])
        return Array(shape, values, self.dtype, False)

    def astype(self, dtype):
        assert dtype == 'float32'
        return Array(self.shape, [f32(v) for v in self.data], dtype, True)

    def __getitem__(self, rows):
        start, stop, step = rows.indices(self.shape[0])
        block = math.prod(self.shape[1:])
        values = [value for row in range(start, stop, step)
                  for value in self.data[row * block:(row + 1) * block]]
        return Array((len(range(start, stop, step)), *self.shape[1:]), values, self.dtype, False)

    def arithmetic(self, other, operation):
        if not isinstance(other, Array):
            other = Array((), [other], 'float32')
        assert other.ndim <= self.ndim
        padded = (1,) * (self.ndim - other.ndim) + other.shape
        assert all(b in (1, a) for a, b in zip(self.shape, padded))
        result = []
        for coordinate, value in zip(coordinates(self.shape), self.data):
            right = tuple(0 if n == 1 else i for i, n in zip(coordinate, padded))
            right = right[-other.ndim:] if other.ndim else ()
            result.append(f32(operation(value, other.data[other.offset(right)])))
        return Array(self.shape, result, 'float32', True)

    def __mul__(self, other):
        return self.arithmetic(other, lambda a, b: a * b)

    def __sub__(self, other):
        return self.arithmetic(other, lambda a, b: a - b)

    def __truediv__(self, other):
        return self.arithmetic(other, lambda a, b: a / b)


def fake_np(events):
    def asarray(value, dtype=None):
        if isinstance(value, Array):
            return value
        shape = () if isinstance(value, (int, float)) else (len(value),)
        data = [value] if not shape else value
        return Array(shape, [f32(v) for v in data] if dtype == 'float32' else data, dtype)

    def inplace(label, operation):
        def apply(value, other, *, out):
            assert out is value and value.owned
            value.data = value.arithmetic(other, operation).data
            events.append((label, id(out)))
        return apply

    def concatenate(values):
        first = values[0]
        return Array((sum(value.shape[0] for value in values), *first.shape[1:]),
                     [item for value in values for item in value.data], first.dtype, True)

    def repeat(value, count, axis):
        assert axis == 0 and value.shape[0] == 1
        return Array((count, *value.shape[1:]), value.data * count, value.dtype, True)

    return SimpleNamespace(asarray=asarray, float32=f32, uint8='uint8',
                           multiply=inplace('multiply', lambda a, b: a * b),
                           subtract=inplace('subtract', lambda a, b: a - b),
                           divide=inplace('divide', lambda a, b: a / b),
                           ascontiguousarray=lambda a: a, concatenate=concatenate, repeat=repeat)


def method(fallback, np):
    tree = ast.parse((ROOT / 'src/tensorfold/vision/qwen_processing.py').read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'QwenImageProcessor')
    node = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_video_patches')
    if fallback:
        node.body.remove(next(n for n in node.body if isinstance(n, ast.If)
                              and 'frames.dtype' in ast.unparse(n.test)))
    namespace = {'np': np}
    module = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    exec(compile(module, 'current_video_source', 'exec', flags=__import__('__future__').annotations.compiler_flag), namespace)
    return tree, node, namespace['_video_patches']


class Controls(unittest.TestCase):
    def test_odd_padding_and_dtype_or_high_rank_broadcast_against_complete_coordinates(self):
        for mode in ('uint8', 'float16', 'higher-rank'):
            events = []
            np = fake_np(events)
            class Scalar(float):
                ndim = 0
            class Float32(str):
                def __call__(self, value):
                    return Scalar(f32(value))
            np.float32 = Float32('float32')
            _, _, current = method(False, np)
            count, height, width, channels = 3, 4, 4, 3
            source = Array((count, height, width, channels), range(144),
                           dtype='float16' if mode == 'float16' else 'uint8')
            before = list(source.data)
            means, stds = [f32(.1), f32(.2), f32(.3)], [f32(.7), f32(.8), f32(.9)]
            mean = Array((1, 1, 1, 3), means, 'float32') if mode == 'higher-rank' else means
            std = Array((1, 1, 1, 3), stds, 'float32') if mode == 'higher-rank' else stds
            owner = SimpleNamespace(config={'vision_config': {'patch_size': 2,
                'spatial_merge_size': 2, 'temporal_patch_size': 2, 'in_channels': 3}},
                processor=SimpleNamespace(image_mean=mean, image_std=std, rescale_factor=1 / 255))
            actual, grid = current(owner, SimpleNamespace(frames=source))
            expected = []
            for gt, mh, mw, c, t, ph, pw in itertools.product(range(2), range(2), range(2),
                                                            range(3), range(2), range(2), range(2)):
                frame = min(count - 1, gt * 2 + t)
                raw = source.data[source.offset((frame, mh * 2 + ph, mw * 2 + pw, c))]
                expected.append(f32(f32(f32(f32(raw) * f32(1 / 255)) - means[c]) / stds[c]))
            self.assertEqual(actual.data, expected)
            self.assertEqual(actual.shape, (8, 24))
            self.assertEqual(grid.data, [2, 2, 2])
            self.assertEqual(source.data, before)
            self.assertEqual([label for label, _ in events],
                             ['multiply', 'subtract', 'divide'] if mode == 'uint8' else [])

    def test_fast_region_guards_scalar_scale_and_owned_uint8_normalization(self):
        _, node, _ = method(False, fake_np([]))
        added = next(n for n in node.body if isinstance(n, ast.If) and 'frames.dtype' in ast.unparse(n.test))
        test = ast.unparse(added.test)
        for guard in ('frames.dtype == np.uint8', 'scale.ndim == 0', 'mean.ndim <= 1',
                      'std.ndim <= 1', 'mean.size in (1, channels)', 'std.size in (1, channels)'):
            self.assertIn(guard, test)
        sequence = [n.value.func.attr for n in added.body if isinstance(n, ast.Expr)
                    and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Attribute)]
        self.assertEqual(sequence, ['multiply', 'subtract', 'divide'])

    def test_full_method_matches_independent_patch_coordinates(self):
        tested = 0
        for p, merge, temporal, channels, count, h, w in (
                (2, 2, 2, 3, 4, 8, 4), (2, 1, 2, 1, 2, 4, 8), (1, 2, 1, 3, 3, 4, 4)):
            for vector in (False, True):
                events = []
                np = fake_np(events)
                # astype expects NumPy's dtype token; float32 constructor is also callable.
                class Scalar(float):
                    ndim = 0

                class Float32(str):
                    def __call__(self, value):
                        return Scalar(f32(value))
                np.float32 = Float32('float32')
                _, _, old = method(True, np)
                _, _, new = method(False, np)
                shape = (count, h, w, channels)
                source = Array(shape, (i % 256 for i in range(math.prod(shape))))
                before = list(source.data)
                mean = [.1 + c / 10 for c in range(channels)] if vector else .1
                std = [.7 + c / 10 for c in range(channels)] if vector else .7
                owner = SimpleNamespace(config={'vision_config': {'patch_size': p,
                    'spatial_merge_size': merge, 'temporal_patch_size': temporal, 'in_channels': channels}},
                    processor=SimpleNamespace(image_mean=mean, image_std=std, rescale_factor=1 / 255))
                old_value, old_grid = old(owner, SimpleNamespace(frames=source))
                new_value, new_grid = new(owner, SimpleNamespace(frames=source))
                expected = []
                for gt, hm, wm, mh, mw, c, t, ph, pw in itertools.product(
                        range(count // temporal), range(h // p // merge), range(w // p // merge),
                        range(merge), range(merge), range(channels), range(temporal), range(p), range(p)):
                    raw = source.data[source.offset((gt * temporal + t, (hm * merge + mh) * p + ph,
                                                     (wm * merge + mw) * p + pw, c))]
                    a, b = (mean[c], std[c]) if vector else (mean, std)
                    expected.append(f32(f32(f32(raw) * f32(1 / 255)) - f32(a)) / f32(b))
                expected = [f32(v) for v in expected]
                self.assertEqual(old_value.data, expected)
                self.assertEqual(new_value.data, expected)
                self.assertEqual(old_value.shape, new_value.shape)
                self.assertEqual(old_grid.data, new_grid.data)
                self.assertEqual(source.data, before)
                self.assertEqual([label for label, _ in events], ['multiply', 'subtract', 'divide'])
                self.assertEqual(len({identity for _, identity in events}), 1)
                tested += 1
        self.assertEqual(tested, 6)

    def test_non_scalar_scale_keeps_original_broadcast_path(self):
        events = []
        np = fake_np(events)

        class Float32(str):
            def __call__(self, value):
                if isinstance(value, list):
                    return Array((len(value),), [f32(v) for v in value], 'float32')
                raise AssertionError('this control requires non-scalar scale')

        np.float32 = Float32('float32')
        _, _, original = method(True, np)
        _, _, candidate = method(False, np)
        source = Array((2, 4, 4, 3), range(96))
        before = list(source.data)
        for scale in ([1 / 255], [1 / 255, 1 / 257, 1 / 259]):
            owner = SimpleNamespace(config={'vision_config': {'patch_size': 2,
                'spatial_merge_size': 2, 'temporal_patch_size': 2, 'in_channels': 3}},
                processor=SimpleNamespace(image_mean=[.1, .2, .3], image_std=[.7, .8, .9], rescale_factor=scale))
            old, grid = original(owner, SimpleNamespace(frames=source))
            new, new_grid = candidate(owner, SimpleNamespace(frames=source))
            self.assertEqual(old.data, new.data)
            self.assertEqual(grid.data, new_grid.data)
            self.assertEqual(events, [])
            self.assertEqual(source.data, before)


if __name__ == '__main__':
    unittest.main()

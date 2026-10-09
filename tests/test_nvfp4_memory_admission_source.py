"""Strict stdlib controls for the header-only byte bound; no runtime math claim."""
import ast
import builtins
from functools import partial
import importlib.util
from pathlib import Path
import re
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("memory_candidate", ROOT / "src/tensorfold/families/qwen3_5/cuda/nvfp4_memory.py")
memory = importlib.util.module_from_spec(spec)
spec.loader.exec_module(memory)
capacity_tree = ast.parse((ROOT / 'src/tensorfold/cuda/capacity.py').read_bytes())
SIZES = ast.literal_eval(next(n.value for n in capacity_tree.body if isinstance(n, ast.Assign)
                             and any(isinstance(t, ast.Name) and t.id == 'SIZES' for t in n.targets)))


def tensor(dtype, *shape):
    return {"dtype": dtype, "shape": list(shape)}


def projection(n, k):
    return {"m.weight_packed": tensor("U8", n, k // 2),
            "m.weight_scale": tensor("F8_E4M3", n, k // 16),
            "m.weight_global_scale": tensor("F32", 1),
            "m.input_global_scale": tensor("F32", 1)}


class Contracts(unittest.TestCase):
    def setUp(self):
        ordinary = builtins.__import__
        def guard(name, *args, **kwargs):
            if name == 'tensorfold.cuda.capacity':
                return NS(SIZES=SIZES)
            if name.split('.')[0] in {'torch', 'numpy', 'triton', 'mlx', 'tensorfold', 'cuda', 'ctypes', 'cupy'}:
                raise AssertionError('actual SDK/native import forbidden: '+name)
            return ordinary(name, *args, **kwargs)
        owner = patch.object(builtins, '__import__', guard)
        owner.start()
        self.addCleanup(owner.stop)

    def test_pack_bound_all_loop_sizes(self):
        for n in (1, 63, 64, 127, 128, 4095, 4096, 4097, 8192, 8193):
            for k in (64, 4096, 12288):
                npad = -(-n // 128) * 128
                c = min(npad, 4096)
                raw = n * k // 2 + n * k // 16 + 8
                exact_formula = raw + npad * k // 2 + 4 * npad * k // 16 + 6 * npad * k // 64 + 32 * c * k + (4 << 20)
                self.assertEqual(memory.loading_bytes(projection(n, k)), exact_formula)
                # Independent source lifetime maxima in bytes per qmm cell:
                # block=.5, q/picked=4, packed=.5, widened codes/shift=8.
                points = [9.5, 17, 13, 25, 12.625]
                self.assertGreaterEqual(32 * c * k, max(points) * c * k)

    def test_head_alias_and_conversions(self):
        for stem in ("lm_head", "model.language_model.embed_tokens"):
            self.assertEqual(memory.loading_bytes({stem + ".weight": tensor("BF16", 248320, 4096)}), 0)
            self.assertEqual(memory.loading_bytes({stem + ".weight": tensor("F32", 8, 64)}), 6 * 8 * 64)
        self.assertEqual(memory.loading_bytes({"m.input_layernorm.weight": tensor("BF16", 4096)}), 12 * 4096)
        self.assertEqual(memory.loading_bytes({"m.input_layernorm.weight": tensor("F64", 4096)}), 24 * 4096)
        self.assertEqual(memory.loading_bytes({"m.input_layernorm.weight": tensor("F8_E4M3", 4096)}), 11 * 4096)
        self.assertEqual(memory.loading_bytes({"m.conv1d.weight": tensor("U8", 16, 1, 4)}), 11 * 16 * 4)
        for dtype, size in SIZES.items():
            for name in ('m.conv1d.weight', 'm.norm.weight', 'm.input_layernorm.weight',
                         'm.post_attention_layernorm.weight', 'm.q_norm.weight', 'm.k_norm.weight'):
                self.assertEqual(memory.loading_bytes({name: tensor(dtype, 16, 4)}),
                                 (size + 8 + max(2, size)) * 16 * 4)
        self.assertEqual(memory.loading_bytes({"m.A_log": tensor("F32", 16)}), 12 * 16)
        self.assertEqual(memory.loading_bytes({"m.weight_global_scale": tensor("F64", 1)}), 16)
        self.assertEqual(memory.loading_bytes({"m.input_global_scale": tensor("I64", 1)}), 16)
        for dtype, size in SIZES.items():
            expected = 0 if dtype == 'BF16' else (size + 2) * 128 * 64
            self.assertEqual(memory.loading_bytes({'model.embed_tokens.weight': tensor(dtype, 128, 64)}), expected)

    def test_skips_and_maximum_not_sum(self):
        part = projection(4096, 12288)
        peak = memory.loading_bytes(part)
        other = {"n." + n: i for n, i in part.items()}
        self.assertEqual(memory.loading_bytes(part | other), peak)
        skipped = {"model.visual." + n: i for n, i in projection(248320, 4096).items()}
        self.assertEqual(memory.loading_bytes(part | skipped), peak)

    def test_schedule_excludes_proved_late_storage(self):
        def transform(name, info):
            count = 1
            for n in info["shape"]:
                count *= n
            return count * {"BF16": 2, "F32": 4, "U8": 1, "F8_E4M3": 1}[info["dtype"]], 0
        layer = {"model.layers.0." + n: i for n, i in projection(4096, 12288).items()}
        dense = {"model.embed_tokens.weight": tensor("BF16", 248320, 4096),
                 "lm_head.weight": tensor("BF16", 248320, 4096),
                 "model.norm.weight": tensor("BF16", 4096)}
        self.assertEqual(memory.scheduled_loading_bytes(layer | dense, transform), 4 << 20)
        small = {"model.embed_tokens.weight": tensor("BF16", 128, 64),
                 "lm_head.weight": tensor("BF16", 128, 64)}
        envelope = memory.loading_bytes(layer)
        self.assertEqual(memory.scheduled_loading_bytes(layer | small, transform), envelope - 2 * 128 * 64 * 2 + (4 << 20))
        with self.assertRaises(ValueError):
            memory.scheduled_loading_bytes(layer, transform)

    def test_refuses_unsupported_projection(self):
        for header in ({"m.weight": tensor("U8", 128, 32)},
                       {"m.weight": tensor("I32", 128, 8)}, projection(128, 66),
                       {"m.weight": tensor("F8_E4M3", 128, 65)}):
            with self.assertRaises(ValueError):
                memory.loading_bytes(header)

    def test_actual_engine_selects_only_the_supported_loading_region(self):
        source = ROOT / 'src/tensorfold/families/qwen3_5/cuda/engine.py'
        tree = ast.parse(source.read_bytes())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Qwen27Engine')
        constructor = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '__init__')
        branch = next(n for n in ast.walk(constructor) if isinstance(n, ast.If) and ast.unparse(n.test) == 'nvfp4')
        transform, geometry = object(), object()
        ordinary = builtins.__import__
        def imports(name, *args, **kwargs):
            if name == 'nvfp4_load':
                return NS(admission=lambda value: (value, transform))
            if name == 'nvfp4_memory':
                return memory
            return ordinary(name, *args, **kwargs)
        for fp8 in (False, True):
            for unified in (False, True):
                for draft in (None, 'owned-draft-directory'):
                    scope = dict(geometry=geometry, prompt_precision=NS(fp8=lambda: fp8),
                                 unified=lambda _: unified, torch=object(), draft_dir=draft,
                                 weight_estimator=None, partial=partial)
                    with patch.object(builtins, '__import__', imports):
                        exec(compile(ast.fix_missing_locations(ast.Module(branch.body, [])), str(source), 'exec'), scope)
                    selected = not fp8 and not unified and draft is None
                    if selected:
                        self.assertIs(scope['weight_estimator'].func, memory.weight_estimate)
                        self.assertIs(scope['weight_estimator'].keywords['transform'], transform)
                    else:
                        self.assertIsNone(scope['weight_estimator'])

    def test_header_estimate_retains_original_host_budget_and_corrects_device_overlap(self):
        entries = {"model.layers.0." + name: info for name, info in projection(4096, 12288).items()}
        entries.update({"model.embed_tokens.weight": tensor("BF16", 248320, 4096),
                        "lm_head.weight": tensor("BF16", 248320, 4096),
                        "model.norm.weight": tensor("BF16", 4096)})
        sizes = SIZES
        class Weights:
            def __init__(self, resident, staging, mapped):
                self.resident, self.staging, self.mapped = resident, staging, mapped
        def transform(name, info):
            if name.startswith(('model.visual.', 'visual.', 'vision_tower', 'mtp.')) or '.visual.' in name or '.mtp.' in name:
                return 0, 0
            shape = list(info['shape'])
            if name.endswith(('.A_log', '.dt_bias')):
                return memory.math.prod(shape)*4, 0
            if len(shape) == 2 and info['dtype'] in ('U8', 'F8_E4M3') and not name.endswith('_scale'):
                shape[0] = -(-shape[0]//128)*128
            return memory.math.prod(shape)*sizes[info['dtype']], 0
        def generic(values, change):
            resident = largest = 0
            layers = {}
            for name, info in values.items():
                size, _ = change(name, info)
                resident += size
                largest = max(largest, size)
                match = re.search(r'(?:layers|blocks)\.(\d+)\.', name)
                key = match.group(1) if match else name
                layers[key] = layers.get(key, 0)+size
            return Weights(resident, 3*max([largest, *layers.values()]), 0)
        capacity = NS(SIZES=SIZES, Weights=Weights, _estimate_weights_from_headers=generic, headers=lambda _: entries,
                      itemsize=lambda info, name: sizes[info['dtype']])
        ordinary = builtins.__import__
        def imports(name, *args, **kwargs):
            if name == 'tensorfold.cuda.capacity':
                return capacity
            return ordinary(name, *args, **kwargs)
        with patch.object(builtins, '__import__', imports):
            weight, host = memory.weight_estimate(Path('opaque-model'), transform)
        packed = 4096 * 12288 // 2
        scales = 4096 * 12288 // 16
        dense = 2 * 248320 * 4096 * 2
        self.assertEqual((weight.resident, weight.staging, weight.mapped),
                         (packed + scales + 8 + dense + 8192, 4194304, 0))
        self.assertEqual(host, 6102712320)
        # Full math pads the block-scale plane independently of packed words.
        # A nonaligned row count must reserve both planes' retained padding.
        entries['model.layers.0.m.weight_packed']['shape'][0] = 4097
        entries['model.layers.0.m.weight_scale']['shape'][0] = 4097
        with patch.object(builtins, '__import__', imports):
            padded, padded_host = memory.weight_estimate(Path('opaque-model'), transform)
        self.assertEqual(padded.resident, 4224 * 12288 // 2 + 4224 * 12288 // 16 + 8 + dense + 8192)
        self.assertEqual(padded_host, host)
        # Skipped vision/MTP payloads never enter the host read buffers.
        entries['model.visual.weight'] = tensor('BF16', 16_000_000_000)
        with patch.object(builtins, '__import__', imports):
            skipped, skipped_host = memory.weight_estimate(Path('opaque-model'), transform)
        self.assertEqual((skipped.resident, skipped.staging, skipped.mapped),
                         (padded.resident, padded.staging, padded.mapped))
        self.assertEqual(skipped_host, host)
        # The loader casts embedding storage to BF16 independently of the
        # projection format. One-byte stored embeddings need resident expansion.
        del entries['model.visual.weight']
        for dtype in ('U8', 'F8_E4M3', 'I8', 'BOOL'):
            entries['model.embed_tokens.weight']['dtype'] = dtype
            with patch.object(builtins, '__import__', imports):
                converted, _ = memory.weight_estimate(Path('opaque-model'), transform)
            self.assertEqual(converted.resident, padded.resident)

    def test_nvfp4_scale_geometry_refused_before_sizing(self):
        entries = projection(129, 4096)
        entries['m.weight_scale']['shape'] = [128, 256]
        with self.assertRaisesRegex(ValueError, 'block scales must have shape'):
            memory.loading_bytes(entries)

    def test_loading_schedule_and_pack_geometry_match_owned_source_contract(self):
        loader = ast.parse((ROOT / 'src/tensorfold/families/qwen3_5/cuda/nvfp4_load.py').read_bytes())
        function = next(n for n in loader.body if isinstance(n, ast.FunctionDef) and n.name == 'load_nvfp4')
        block = next(n for n in function.body if isinstance(n, ast.Try))
        loop_index = next(i for i, n in enumerate(block.body) if isinstance(n, ast.For))
        weight_index = next(i for i, n in enumerate(block.body)
                            if isinstance(n, ast.Assign) and isinstance(n.value, ast.Call)
                            and ast.unparse(n.value.func) == 'Weights')
        self.assertLess(loop_index, weight_index)
        weight_call = block.body[weight_index].value
        self.assertEqual([n.arg for n in weight_call.keywords],
                         ['config', 'embed', 'layers', 'norm', 'head', 'quant'])
        self.assertEqual(ast.unparse(weight_call.keywords[1].value),
                         "Plain(get('embed_tokens.weight').to(torch.bfloat16))")
        self.assertEqual(ast.unparse(weight_call.keywords[3].value), "norm('norm.weight')")
        self.assertEqual(ast.unparse(weight_call.keywords[4].value), "linear('lm_head', prompt=False)")
        pack = ast.parse((ROOT / 'src/tensorfold/cuda/kernels/qmm.py').read_bytes())
        pack_function = next(n for n in pack.body if isinstance(n, ast.FunctionDef) and n.name == 'pack')
        self.assertEqual(pack_function.args.args[-1].arg, 'chunk')
        self.assertEqual(ast.literal_eval(pack_function.args.defaults[-1]), 4096)


if __name__ == "__main__":
    unittest.main()

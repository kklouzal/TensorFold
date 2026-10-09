"""Actual FP8 startup source gates; labeled metadata, no SDK/native execution."""
from __future__ import annotations

import ast
from contextlib import contextmanager
import importlib.abc
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / 'src/tensorfold/families/qwen3_5/cuda/weights.py'
NVFP4 = WEIGHTS.with_name('nvfp4_load.py')


class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'numpy', 'triton', 'mlx', 'cupy', 'cuda', 'ctypes'}:
            raise AssertionError('local SDK/native import forbidden: ' + fullname)


def config(**changes):
    cls = next(node for node in ast.parse(WEIGHTS.read_text()).body
               if isinstance(node, ast.ClassDef) and node.name == 'Config')
    cls = ast.ClassDef(name='Config', bases=[], keywords=[], body=cls.body, decorator_list=[])
    scope = {'Path': Path, '__name__': __name__}
    future = ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)
    exec(compile(ast.fix_missing_locations(ast.Module([future, cls], [])), str(WEIGHTS), 'exec'), scope)
    fields = dict(hidden=256, vocab=128, layers=4, interval=4, experts=0, head_dim=128,
                  rope_dims=32, heads=4, kv_heads=2, k_heads=2, v_heads=4, dk=128,
                  dv=128, conv_kernel=4, top_k=0, moe_width=0, intermediate=512)
    fields.update(changes)
    # Large-head cap witnesses must satisfy the existing Config04 contracts,
    # rather than being rejected first for query/KV ratio or GDN grouping.
    if 'heads' in changes and 'kv_heads' not in changes:
        fields['kv_heads'] = fields['heads']
    if 'v_heads' in changes and 'k_heads' not in changes:
        fields['k_heads'] = 1
    value = scope['Config']()
    value.__dict__.update(fields)
    return value


def source_scope(path, names, **scope):
    nodes = [node for node in ast.parse(path.read_text()).body
             if isinstance(node, ast.FunctionDef) and node.name in names]
    future = ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)
    exec(compile(ast.fix_missing_locations(ast.Module([future, *nodes], [])), str(path), 'exec'), scope)
    return scope


def entries(cfg, *, dense=True, dtype='BF16'):
    result = {}
    for i in range(cfg.layers):
        base = f'model.layers.{i}.'
        names = [base + 'mlp.' + name for name in ('gate_proj', 'up_proj', 'down_proj')] if dense else []
        names += ([base + 'linear_attn.' + name for name in
                   ('in_proj_qkv', 'in_proj_z', 'in_proj_b', 'in_proj_a', 'out_proj')]
                  if cfg.is_linear(i) else
                  [base + 'self_attn.' + name for name in ('q_proj', 'k_proj', 'v_proj', 'o_proj')])
        for name in names:
            for suffix in ('.weight', '.scales', '.biases'):
                result[name + suffix] = (Path('/labeled-header'), 0, 0, dtype, [1, 1])
    return result


@contextmanager
def imports(*, fp8=True, spec=None, exl3=False, nvfp4=False, own=False, headers=None):
    modules = {}
    def module(name, **values):
        result = ModuleType(name)
        result.__dict__.update(values)
        modules[name] = result
        return result
    module('tensorfold')
    module('tensorfold.cuda', prompt_precision=SimpleNamespace(fp8=lambda: fp8))
    module('tensorfold.quantization', resolve_affine=lambda raw, name: spec, validate_shapes=lambda *args: None)
    module('tensorfold.families')
    module('tensorfold.families.qwen3_5')
    module('tensorfold.families.qwen3_5.cuda')
    module('tensorfold.families.qwen3_5.cuda.exl3_load', quant_config=lambda path: {} if exl3 else None,
           load_exl3=lambda *a: 'EXL3 selected')
    module('tensorfold.families.qwen3_5.cuda.nvfp4_load', quantized=lambda path: nvfp4,
           load_nvfp4=lambda *a: 'NVFP4 selected')
    module('triton')
    module('triton.language')
    module('triton.language.core', TRITON_MAX_TENSOR_NUMEL=1 << 20)
    module('tensorfold.cuda.capacity', headers=lambda path: headers or {})
    module('tensorfold.cuda.nvfp4', format=SimpleNamespace(config_block=lambda raw: {}, scheme=lambda parts: 'bf16'))
    module('tensorfold.cuda.nvfp4.linear', Fp4Linear=object, Fp8Linear=object, Staging=lambda: object())
    with patch.dict(sys.modules, modules):
        yield modules


class PayloadReached(RuntimeError):
    pass


class Controls(unittest.TestCase):
    def setUp(self):
        guard = Guard()
        sys.meta_path.insert(0, guard)
        self.addCleanup(lambda: sys.meta_path.remove(guard))

    def test_exact_rounded_products_capacity_and_local_shards(self):
        for field, limit in (('heads', 8192), ('v_heads', 8192)):
            cfg = config(**{field: limit})
            cfg.validate_geometry(max_tensor_numel=1 << 20)
            cfg.validate_fp8_prefill(max_tensor_numel=1 << 20)
            above = config(**{field: limit + 1})
            above.validate_geometry(max_tensor_numel=1 << 20)
            with self.assertRaises(ValueError):
                above.validate_fp8_prefill(max_tensor_numel=1 << 20)
            sharded = config(**{field: 2 * limit, 'k_heads': 2})
            sharded.validate_geometry(max_tensor_numel=1 << 20)
            sharded.validate_fp8_prefill(max_tensor_numel=1 << 20, world=2)
        config(heads=4096, head_dim=256).validate_fp8_prefill(max_tensor_numel=1 << 20)
        with self.assertRaises(ValueError):
            config(heads=4097, head_dim=256).validate_fp8_prefill(max_tensor_numel=1 << 20)
        with self.assertRaises(ValueError):
            config(v_heads=3).validate_fp8_prefill(max_tensor_numel=1 << 20, world=2)
        for bad in (True, 0, 3):
            with self.assertRaises(ValueError):
                config().validate_fp8_prefill(max_tensor_numel=1 << 20, world=bad)

    def test_unused_attention_and_GDN_metadata_remain_legal(self):
        config(layers=0, heads=2**30, v_heads=2**30).validate_fp8_prefill(max_tensor_numel=64)
        config(layers=1, interval=1, v_heads=2**30).validate_fp8_prefill(max_tensor_numel=8192)
        config(layers=1, interval=8, heads=2**30).validate_fp8_prefill(max_tensor_numel=512)

    def test_metadata_selection_matches_existing_fast_predicate(self):
        cfg = config()
        scope = source_scope(WEIGHTS, {'_fp8_prefill_metadata'})
        reader = SimpleNamespace(files=SimpleNamespace(where=entries(cfg)))
        for bits, gs, dtype, expected in ((4, 64, 'BF16', True), (3, 64, 'BF16', False),
                                         (4, 32, 'BF16', False), (4, 64, 'F32', False)):
            reader.files.where = entries(cfg, dtype=dtype)
            with imports(spec=SimpleNamespace(bits=bits, group_size=gs)):
                self.assertIs(scope['_fp8_prefill_metadata'](cfg, {}, reader, '',
                              ('gate_proj', 'up_proj', 'down_proj')), expected)
        with imports(spec=None):
            self.assertFalse(scope['_fp8_prefill_metadata'](cfg, {}, reader, '', ()))
        reader.files.where = {'language_model.' + name: info for name, info in entries(cfg, dense=False).items()}
        with imports(spec=SimpleNamespace(bits=4, group_size=64)):
            self.assertTrue(scope['_fp8_prefill_metadata'](cfg, {}, reader, 'language_model.', ()))

    def run_mlx(self, cfg, *, fp8=True, gs=64, dtype='BF16', world=1, mlp=None, prefill_mlp=None):
        cfg.validate_geometry(max_tensor_numel=1 << 20)
        events = []
        class Reader:
            def __init__(self, *args):
                self.files = SimpleNamespace(where=entries(cfg, dense=prefill_mlp != (), dtype=dtype))
                self.where = dict.fromkeys(self.files.where)
            def __iter__(self):
                return iter(self.where)
            def pop(self, name):
                events.append('payload')
                raise PayloadReached('labeled payload; never read or imported SDK')
            def close(self):
                events.append('close')
        scope = source_scope(WEIGHTS, {'load', '_fp8_prefill_metadata', '_validate_fp8_prefill'},
                             __name__='tensorfold.families.qwen3_5.cuda.weights', __package__='tensorfold.families.qwen3_5.cuda',
                             Path=Path, Config=SimpleNamespace(read=lambda path: cfg), _Tensors=Reader,
                             GDN=SimpleNamespace, Attention=SimpleNamespace,
                             read_metadata_json=lambda path: {}, checkpoint_path=lambda path, name: path / name,
                             _close_failed_checkpoint=lambda reader, error: reader.close())
        with imports(fp8=fp8, spec=SimpleNamespace(bits=4, group_size=gs)):
            with self.assertRaises((ValueError, PayloadReached)) as raised:
                scope['load'](Path('/labeled-model'), prefill_world=world, mlp=mlp, prefill_mlp=prefill_mlp)
        self.assertEqual(events[-1:], ['close'])
        return type(raised.exception), events

    def test_selected_MLX_refuses_before_any_weight_payload(self):
        error, events = self.run_mlx(config(heads=8193))
        self.assertIs(error, ValueError)
        self.assertNotIn('payload', events)
        error, events = self.run_mlx(config(v_heads=8193))
        self.assertIs(error, ValueError)
        self.assertNotIn('payload', events)

    def test_BF16_and_nonfast_formats_keep_their_original_paths(self):
        for changes in ({'fp8': False}, {'gs': 32}, {'dtype': 'F32'}):
            error, events = self.run_mlx(config(heads=8193), **changes)
            self.assertIs(error, PayloadReached)
            self.assertEqual(events, ['payload', 'close'])
        error, _ = self.run_mlx(config(heads=16384, v_heads=16384, k_heads=2), world=2)
        self.assertIs(error, PayloadReached)

    def test_routed_projection_declaration_checks_before_payload(self):
        error, events = self.run_mlx(config(v_heads=8193), mlp=lambda *args: {}, prefill_mlp=())
        self.assertIs(error, ValueError)
        self.assertEqual(events, ['close'])
        # An opaque public callback has no proven header projection set. Its
        # existing construction remains legal; its actual fast_prefill result
        # is checked after construction, before execution.
        error, events = self.run_mlx(config(v_heads=8193), mlp=lambda *args: {})
        self.assertIs(error, PayloadReached)
        self.assertEqual(events, ['payload', 'close'])

    def test_malformed_projection_declarations_stop_before_reader(self):
        scope = source_scope(WEIGHTS, {'load'}, __package__='tensorfold.families.qwen3_5.cuda', Path=Path)
        for names in ([], ['gate_proj'], ('gate_proj', 'gate_proj'), ('weight',), (True,), (1,)):
            with imports(exl3=True):
                with self.assertRaises(ValueError):
                    scope['load'](Path('/labeled-model'), prefill_mlp=names)

    def test_original_fast_properties_agree_with_header_selection(self):
        module = ast.parse(WEIGHTS.read_text())
        methods = {}
        for cls, name in (('QLinear', 'fast'), ('Weights', 'fast_prefill')):
            node = next(item for item in module.body if isinstance(item, ast.ClassDef) and item.name == cls)
            fn = next(item for item in node.body if isinstance(item, ast.FunctionDef) and item.name == name)
            fn.decorator_list = []
            methods[name] = fn
        scope = {'torch': SimpleNamespace(bfloat16='BF16')}
        exec(compile(ast.fix_missing_locations(ast.Module(list(methods.values()), [])), str(WEIGHTS), 'exec'), scope)
        metadata = source_scope(WEIGHTS, {'_fp8_prefill_metadata'})['_fp8_prefill_metadata']
        cfg = config()
        for bits, gs, dtype in ((4, 64, 'BF16'), (3, 64, 'BF16'), (4, 32, 'BF16'), (4, 64, 'F32')):
            q = SimpleNamespace(bits=bits, gs=gs, layout='mlx', scales=SimpleNamespace(dtype=dtype),
                                biases=SimpleNamespace(dtype=dtype))
            q.fast = scope['fast'](q)
            layers = []
            for i in range(cfg.layers):
                gdn = SimpleNamespace(**dict.fromkeys(('qkv', 'z', 'b', 'a', 'out'), q)) if cfg.is_linear(i) else None
                attn = SimpleNamespace(**dict.fromkeys(('q', 'k', 'v', 'o'), q)) if not cfg.is_linear(i) else None
                layers.append(SimpleNamespace(gate=q, up=q, down=q, gdn=gdn, attn=attn))
            weights = SimpleNamespace(quant='mlx', layers=layers)
            reader = SimpleNamespace(files=SimpleNamespace(where=entries(cfg, dtype=dtype)))
            with imports(spec=SimpleNamespace(bits=bits, group_size=gs)):
                selected = metadata(cfg, {}, reader, '', ('gate_proj', 'up_proj', 'down_proj'))
            self.assertIs(selected, scope['fast_prefill'](weights))

    def test_EXL3_delegates_without_FP8_glue_admission(self):
        scope = source_scope(WEIGHTS, {'load'}, __package__='tensorfold.families.qwen3_5.cuda', Path=Path)
        with imports(exl3=True):
            self.assertEqual(scope['load'](Path('/labeled-model')), 'EXL3 selected')

    def test_NV_full_and_checkpoint_routes_before_payload(self):
        for own, fp8, refused in ((False, True, True), (True, True, False), (False, False, False)):
            cfg, events = config(v_heads=8193), []
            class Reader:
                def __init__(self, *args, **kwargs):
                    pass
                def pop(self, name):
                    events.append('payload')
                    raise PayloadReached()
                def close(self):
                    events.append('close')
            helper = source_scope(WEIGHTS, {'_validate_fp8_prefill'})['_validate_fp8_prefill']
            scope = source_scope(NVFP4, {'load_nvfp4'}, __package__='tensorfold.families.qwen3_5.cuda',
                                 Path=Path, skipped=lambda name: False, SUFFIXES=('weight',),
                                 prompt_precision=SimpleNamespace(fp8=lambda: fp8),
                                 read_metadata_json=lambda path: {}, checkpoint_path=lambda path, name: path / name,
                                 maths=lambda: ({'nvfp4': own, 'fp8': own}, 'labeled mode'))
            metadata = {'model.layers.0.linear_attn.in_proj_qkv.weight': {'dtype': 'BF16', 'shape': [1, 1]}}
            with imports(headers=metadata) as modules:
                fake = ModuleType('tensorfold.families.qwen3_5.cuda.weights')
                fake.__dict__.update(GDN=object, Attention=object, Config=SimpleNamespace(read=lambda path: cfg),
                                     Layer=object, Weights=object, _Tensors=Reader,
                                     _close_failed_checkpoint=lambda reader, error: reader.close(),
                                     _validate_fp8_prefill=helper)
                modules[fake.__name__] = fake
                with patch.dict(sys.modules, {fake.__name__: fake}):
                    with self.assertRaises((ValueError, PayloadReached)) as raised:
                        scope['load_nvfp4'](Path('/labeled-model'))
            self.assertIs(type(raised.exception), ValueError if refused else PayloadReached)
            self.assertEqual(events, ['close'] if refused else ['payload', 'close'])


if __name__ == '__main__':
    unittest.main()

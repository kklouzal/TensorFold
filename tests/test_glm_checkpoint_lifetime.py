"""Owned GLM checkpoint reader lifetime/path controls; no SDK execution."""
import ast
import json
from pathlib import Path
import runpy
import struct
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from tensorfold.cuda.tensor_file import checkpoint_path, read_metadata_json  # noqa: E402

WEIGHTS = ROOT / 'src/tensorfold/families/glm5_next/cuda/weights.py'
SPLIT = ROOT / 'src/tensorfold/families/glm5_next/cuda/split.py'


class Controls(unittest.TestCase):
    def load(self, fault=None, cleanup=None):
        calls = []
        primary = KeyboardInterrupt('load interrupted')
        class Tensor:
            weight = None
            def to(self, *args):
                return self
            def contiguous(self):
                return self
            def __getitem__(self, key):
                return self
        tensor = Tensor()
        tensor.weight = tensor
        class Reader:
            def __init__(self, *args):
                calls.append('acquired')
            def get(self, name):
                calls.append(name)
                if fault and fault in name:
                    raise primary
                return tensor
            def close(self):
                calls.append('closed')
                if cleanup:
                    raise cleanup
        class BadHeads:
            def __floordiv__(self, value):
                raise primary
        config = SimpleNamespace(quant='exl3', heads=BadHeads() if fault == 'geometry' else 4,
            lin_heads=4, layers=0, vocab=128, mtp_layers=0)
        provider = ModuleType('owned_glm_load.split')
        provider.RankReader = Reader
        tree = ast.parse(WEIGHTS.read_bytes())
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'load')
        namespace = {'__package__': 'owned_glm_load', 'Config': SimpleNamespace(read=lambda _: config),
            'torch': SimpleNamespace(device=lambda x: x, bfloat16='BF16',
                                     cuda=SimpleNamespace(empty_cache=lambda: calls.append('cache'))),
            'PREFIX': 'model.language_model.', 'make_b16': lambda x: x,
            'quantize4': lambda x: x, 'Weights': lambda *a, **k: SimpleNamespace(meta={})}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(WEIGHTS), 'exec',
                     flags=__import__('__future__').annotations.compiler_flag), namespace)
        with patch.dict(sys.modules, {'owned_glm_load.split': provider}):
            if fault or cleanup:
                with self.assertRaises(BaseException) as caught:
                    namespace['load'](Path('owned'), rank=0)
                self.assertIs(caught.exception, primary if fault else cleanup)
                if fault and cleanup:
                    self.assertIs(primary.__cause__, cleanup)
                    self.assertTrue(primary.__notes__)
            else:
                namespace['load'](Path('owned'), rank=0)
        self.assertEqual(calls.count('closed'), 1)
        self.assertEqual(calls[0], 'acquired')
        if fault or cleanup:
            self.assertNotIn('cache', calls)
        else:
            self.assertEqual(calls[-2:], ['closed', 'cache'])

    def test_success_and_all_acquired_early_or_late_failures_close_once(self):
        self.load()
        for fault in ('geometry', 'embed_tokens', 'lm_head', 'norm.weight'):
            with self.subTest(fault=fault):
                self.load(fault)

    def test_primary_survives_reader_drain_and_success_drain_failure_propagates(self):
        for fault in ('geometry', 'embed_tokens', 'lm_head'):
            with self.subTest(fault=fault):
                self.load(fault, OSError('reader drain failed'))
        self.load(cleanup=OSError('reader drain failed'))

    def split_module(self, calls, cleanup=None):
        class Reader:
            def __init__(self):
                calls.append('reader')
            def close(self):
                calls.append('closed')
                if cleanup:
                    raise cleanup
        class ReadAhead(Reader):
            def __init__(self, owner, *args):
                self.owner = owner
                calls.append('ahead')
        provider = ModuleType('tensorfold.cuda.direct_read')
        provider.Reader, provider.ReadAhead, provider.SafeTensors = Reader, ReadAhead, object
        numpy = ModuleType('numpy')
        with patch.dict(sys.modules, {'numpy': numpy, 'tensorfold.cuda.direct_read': provider}):
            return runpy.run_path(str(SPLIT), run_name='owned_glm_split_control'), provider

    def test_actual_constructor_foreign_index_drain_preserves_primary(self):
        for shard in ('../foreign.safetensors', '/foreign.safetensors'):
            with self.subTest(shard=shard), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': {'tensor': shard}}))
                calls = []
                cleanup = OSError('drain failed')
                ns, provider = self.split_module(calls, cleanup)
                with patch.dict(sys.modules, {'tensorfold.cuda.direct_read': provider}), self.assertRaises(ValueError) as caught:
                    ns['RankReader'](root, 0)
                self.assertIs(caught.exception.__cause__, cleanup)
                self.assertEqual(calls, ['reader', 'ahead', 'closed'])

    def test_strict_header_and_owned_shard_paths_precede_raw_access(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / 'part.safetensors'
            header = {'tensor': {'dtype': 'U32', 'shape': [1], 'data_offsets': [0, 4]}}
            raw = json.dumps(header).encode()
            path.write_bytes(struct.pack('<Q', len(raw)) + raw + bytes(4))
            (root / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': {'tensor': path.name}}))
            calls = []
            ns, provider = self.split_module(calls)
            with patch.dict(sys.modules, {'tensorfold.cuda.direct_read': provider}):
                reader = ns['RankReader'](root, 1)
            self.assertEqual(reader._paths[path.name], path.resolve())
            self.assertEqual(ns['read_header'](path), (header, 8 + len(raw)))
            reader.close()
            path.write_bytes(struct.pack('<Q', 2**63))
            with self.assertRaises(ValueError):
                ns['read_header'](path)
            calls.clear()
            with patch.dict(sys.modules, {'tensorfold.cuda.direct_read': provider}), self.assertRaises(ValueError):
                ns['RankReader'](root, 2)
            self.assertEqual(calls, [])

    def test_config_strict_parse_and_authorized_root_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'config.json').write_text('{"hidden_size":1,"hidden_size":2}')
            with self.assertRaises(ValueError):
                read_metadata_json(checkpoint_path(root, 'config.json'))
        tree = ast.parse(WEIGHTS.read_bytes())
        config = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Config')
        method = next(n for n in config.body if isinstance(n, ast.FunctionDef) and n.name == 'read')
        self.assertIn("read_metadata_json(checkpoint_path(Path(model_dir), 'config.json'))", ast.unparse(method))


if __name__ == '__main__':
    unittest.main()

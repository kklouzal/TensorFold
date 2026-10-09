"""Actual Config/quant reader boundaries; real files, stdlib only, no SDK execution."""
from __future__ import annotations

import __future__
import ast
from dataclasses import asdict, dataclass
import importlib.util
import json
import struct
from pathlib import Path
from types import ModuleType, SimpleNamespace
import tempfile
import os
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src/tensorfold/families/qwen3_5/cuda"
spec = importlib.util.spec_from_file_location("qwen35_config_real_metadata", ROOT / "src/tensorfold/cuda/tensor_file.py")
HEADER = importlib.util.module_from_spec(spec)
spec.loader.exec_module(HEADER)


spec = importlib.util.spec_from_file_location("qwen35_config_python_file_scope", ROOT / "src/tensorfold/file_io.py")
FILE_IO = importlib.util.module_from_spec(spec)
spec.loader.exec_module(FILE_IO)


class PythonDescriptorOwner:
    """Explicit real-file substitute; no native acquisition/C-return qualification."""
    def __init__(self):
        self.file = None

    def open(self, path, flags, mode=0o600):
        if not self.closed:
            raise RuntimeError("owned metadata fixture slot is already open")
        self.file = open(path, "rb", buffering=0, opener=lambda name, unused: os.open(name, flags, mode))

    def fileno(self):
        if self.file is None:
            raise ValueError("owned metadata fixture slot is closed")
        return self.file.fileno()

    @property
    def closed(self):
        return self.file is None or self.file.closed

    def close(self):
        if self.file is not None:
            self.file.close()



def config(path):
    node = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef) and n.name == "Config")
    scope = {"__name__": __name__, "dataclass": dataclass, "Path": Path, "json": json,
             "checkpoint_path": HEADER.checkpoint_path, "read_metadata_json": HEADER.read_metadata_json}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec",
                 flags=__future__.annotations.compiler_flag), scope)
    return scope["Config"]


def format_api(family):
    path = ROOT / f"src/tensorfold/cuda/{family}/format.py"
    names = ({"read_header", "read_scalar", "config_fields", "is_exl3", "scan", "Checkpoint"} if family == "exl3"
             else {"config_block", "is_quantized"})
    nodes = [n for n in ast.parse(path.read_text()).body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    scope = {"__name__": __name__, "dataclass": dataclass, "PARTS": (), "MARKERS": {}, "Path": Path, "struct": struct, "METHODS": ("modelopt", "compressed-tensors"),
             "SIZES": {"I32": 4, "U32": 4, "I64": 8}, "_regular_stream": HEADER._regular_stream,
             "_strict_header": HEADER.read_header, "read_header_stream": HEADER.read_header_stream,
             "checkpoint_path": HEADER.checkpoint_path, "read_metadata_json": HEADER.read_metadata_json}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec",
                 flags=__future__.annotations.compiler_flag), scope)
    return SimpleNamespace(**scope)


def tensor_file(path, entry, payload):
    raw = json.dumps({"marker": entry, "__metadata__": {"format": "pt"}}).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + payload)


def text():
    return {"hidden_size": 256, "intermediate_size": 512, "num_hidden_layers": 2,
            "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 64,
            "vocab_size": 128, "linear_num_key_heads": 2, "linear_num_value_heads": 4,
            "linear_key_head_dim": 128, "linear_value_head_dim": 128,
            "linear_conv_kernel_dim": 4, "eos_token_id": 3}


def reference():
    """Independent expected Config schema for the complete valid test fixture."""
    return {"hidden": 256, "intermediate": 512, "layers": 2, "heads": 4, "kv_heads": 2,
            "head_dim": 64, "vocab": 128, "k_heads": 2, "v_heads": 4, "dk": 128, "dv": 128,
            "conv_kernel": 4, "interval": 4, "eps": 1e-6, "rope_dims": 16, "rope_theta": 10000000.0,
            "eos": (3,), "experts": 0, "top_k": 0, "moe_width": 0, "mrope_section": (11, 11, 10)}


class ConfigMetadataControls(unittest.TestCase):
    def setUp(self):
        self.owner_patch = patch.object(FILE_IO, "_owned_slot", PythonDescriptorOwner)
        self.module_patch = patch.dict(sys.modules, {"tensorfold.file_io": FILE_IO})
        self.owner_patch.start()
        self.module_patch.start()
        self.addCleanup(self.owner_patch.stop)
        self.addCleanup(self.module_patch.stop)
        self.provider = ModuleType('triton.language.core')
        self.provider.TRITON_MAX_TENSOR_NUMEL = 1 << 20
        provider_modules = {'triton': ModuleType('triton'), 'triton.language': ModuleType('triton.language'),
                            'triton.language.core': self.provider}
        self.provider_patch = patch.dict(sys.modules, provider_modules)
        self.provider_patch.start()
        self.addCleanup(self.provider_patch.stop)

    def test_active_geometry_refusal_uses_actual_selected_provider_before_weight_io(self):
        cls = config(SOURCE / 'weights.py')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            item = text()
            item.update(num_hidden_layers=4, head_dim=128)
            for changes in ({'head_dim': 64}, {'head_dim': 512}, {'num_attention_heads': 64},
                            {'linear_value_head_dim': 64}, {'hidden_size': 32}, {'full_attention_interval': 0}):
                value = dict(item, **changes)
                (root / 'config.json').write_text(json.dumps(value), encoding='utf8')
                with self.assertRaises(ValueError):
                    cls.read(root)
            (root / 'config.json').write_text(json.dumps(item), encoding='utf8')
            self.assertEqual(cls.read(root).head_dim, 128)
            self.provider.TRITON_MAX_TENSOR_NUMEL = 64 * 128 - 1
            with self.assertRaises(ValueError):
                cls.read(root)
            self.provider.TRITON_MAX_TENSOR_NUMEL = 64 * 128
            self.assertEqual(cls.read(root).head_dim, 128)

    def test_valid_outer_nested_generation_and_null_rope_match_original(self):
        new = config(SOURCE / "weights.py")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for nested in (False, True):
                for rope in (None, {}, {"partial_rotary_factor": .5, "rope_theta": 10000,
                                        "mrope_section": [3, 2, 3]}):
                    item = text()
                    item["rope_parameters"] = rope
                    raw = {"text_config": item, "eos_token_id": [3, 7]} if nested else item
                    (root / "config.json").write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
                    (root / "generation_config.json").write_text('{"eos_token_id":[7,9]}', encoding="utf-8")
                    expected = reference()
                    expected["eos"] = (3, 7, 9)
                    if rope:
                        expected.update(rope_dims=32, rope_theta=10000.0, mrope_section=(3, 2, 3))
                    self.assertEqual(asdict(new.read(root)), expected)

    def test_missing_generation_preserves_original_defaults_and_coercions(self):
        new = config(SOURCE / "weights.py")
        item = text()
        item.update(num_hidden_layers=2.0, eos_token_id="3", intermediate_size="512")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text(json.dumps(item), encoding="utf-8")
            self.assertEqual(asdict(new.read(root)), reference())

    def test_config_and_generation_use_strict_metadata_grammar_before_config_use(self):
        cls = config(SOURCE / "weights.py")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for raw in (b'{"hidden_size":2,"hidden_size":3}', b'[]', b'{"value":NaN}', b'{"value":1e999}',
                        b'{"value":"\\ud800"}', b'\xff'):
                (root / "config.json").write_bytes(raw)
                with self.assertRaises((ValueError, UnicodeError)):
                    cls.read(root)
            (root / "config.json").write_text(json.dumps(text()), encoding="utf-8")
            (root / "generation_config.json").write_text('{"eos_token_id":3,"eos_token_id":7}', encoding="utf-8")
            with self.assertRaises(ValueError):
                cls.read(root)

    def test_nested_config_objects_are_checked_before_field_use(self):
        cls = config(SOURCE / "weights.py")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for value in (False, 0, [], "text"):
                (root / "config.json").write_text(json.dumps({"text_config": value}), encoding="utf-8")
                with self.assertRaises(ValueError):
                    cls.read(root)
                item = text()
                item["rope_parameters"] = value
                (root / "config.json").write_text(json.dumps(item), encoding="utf-8")
                with self.assertRaises(ValueError):
                    cls.read(root)

    def test_fixed_config_symlink_cannot_read_an_unrelated_external_file(self):
        cls = config(SOURCE / "weights.py")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model"
            model.mkdir()
            external = root / "other.json"
            external.write_text(json.dumps(text()), encoding="utf-8")
            (model / "config.json").symlink_to(external)
            with self.assertRaises(ValueError):
                cls.read(model)

    def test_shared_quant_detectors_preserve_valid_outer_nested_and_sidecar(self):
        exl, nv = format_api("exl3"), format_api("nvfp4")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for method in ("exl3", "modelopt", "compressed-tensors", "other"):
                for nested in (False, True):
                    raw = {"quantization_config": {"quant_method": method}}
                    if nested:
                        raw = {"text_config": raw}
                    (root / "config.json").write_text(json.dumps(raw), encoding="utf-8")
                    self.assertEqual(exl.is_exl3(root), method == "exl3")
                    self.assertEqual(nv.is_quantized(root), method in ("modelopt", "compressed-tensors"))
            (root / "config.json").unlink()
            (root / "quantization_config.json").write_text('{"quant_method":"EXL3"}', encoding="utf-8")
            self.assertTrue(exl.is_exl3(root))
            self.assertFalse(nv.is_quantized(root))
            (root / "quantization_config.json").write_text('{"quant_method":"exl3","quant_method":"other"}', encoding="utf-8")
            with self.assertRaises(ValueError):
                exl.is_exl3(root)

    def test_shared_exl3_header_and_marker_keep_exact_unsigned_bits(self):
        api = format_api("exl3")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.safetensors"
            for dtype in ("I32", "U32"):
                for shape in ([], [1]):
                    for value in (0, 2**31 - 1, 0xCBAC1FED, 0x83DCD12D, 2**32 - 1):
                        entry = {"dtype": dtype, "shape": shape, "data_offsets": [0, 4]}
                        tensor_file(path, entry, struct.pack("<I", value))
                        self.assertEqual(api.read_header(path), {"marker": entry})
                        self.assertEqual(api.read_scalar(path, entry), value)
            forged = {"dtype": "I32", "shape": [1], "data_offsets": [4, 8]}
            with self.assertRaises(ValueError):
                api.read_scalar(path, forged)
            wide = {"dtype": "I64", "shape": [1], "data_offsets": [0, 8]}
            tensor_file(path, wide, b"12345678")
            with self.assertRaises(ValueError):
                api.read_scalar(path, wide)
            path.write_bytes(b"short")
            with self.assertRaises(ValueError):
                api.read_header(path)

    def test_shared_scan_refuses_duplicate_identity_and_batch_authorizes_paths_before_read(self):
        api = format_api("exl3")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model"
            model.mkdir()
            entry = {"dtype": "I32", "shape": [1], "data_offsets": [0, 4]}
            tensor_file(model / "a.safetensors", entry, b"1234")
            tensor_file(model / "b.safetensors", entry, b"1234")
            with self.assertRaisesRegex(ValueError, "duplicate checkpoint tensor"):
                api.scan(model)
            (model / "b.safetensors").unlink()
            foreign = root / "foreign.safetensors"
            tensor_file(foreign, entry, b"4321")
            (model / "b.safetensors").symlink_to(foreign)
            calls = []
            api.scan.__globals__["read_header"] = lambda path: calls.append(path) or {"unexpected": entry}
            with self.assertRaises(ValueError):
                api.scan(model)
            self.assertEqual(calls, [])

    def test_shared_detectors_authorize_hf_blobs_and_refuse_foreign_symlinks(self):
        exl, nv = format_api("exl3"), format_api("nvfp4")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = root / "models--owner--model"
            model, blobs = store / "snapshots" / "revision", store / "blobs"
            model.mkdir(parents=True)
            blobs.mkdir()
            blob = blobs / "config"
            blob.write_text('{"quantization_config":{"quant_method":"modelopt"}}', encoding="utf-8")
            (model / "config.json").symlink_to(blob)
            self.assertTrue(nv.is_quantized(model))
            self.assertFalse(exl.is_exl3(model))
            (model / "config.json").unlink()
            external = root / "foreign.json"
            external.write_text(blob.read_text(), encoding="utf-8")
            (model / "config.json").symlink_to(external)
            for function in (exl.is_exl3, nv.is_quantized):
                with self.assertRaises(ValueError):
                    function(model)

    def test_every_reopened_family_configuration_uses_shared_reader(self):
        for path in [SOURCE / name for name in ("weights.py", "nvfp4_load.py", "exl3_load.py")] + [ROOT / f"src/tensorfold/cuda/{family}/format.py" for family in ("exl3", "nvfp4")]:
            tree = ast.parse(path.read_text())
            self.assertFalse(any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                                 and n.func.attr in ("loads", "load", "read_text")
                                 and (n.func.attr == "read_text" or isinstance(n.func.value, ast.Name)
                                      and n.func.value.id == "json") for n in ast.walk(tree)))
            calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                     and isinstance(n.func, ast.Name) and n.func.id == "read_metadata_json"]
            self.assertGreaterEqual(len(calls), 1)
            for call in calls:
                self.assertIsInstance(call.args[0], ast.Call)
                self.assertEqual(call.args[0].func.id, "checkpoint_path")


if __name__ == "__main__":
    unittest.main()

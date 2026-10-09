"""Real filesystem admission plus current loader source; no MLX/SDK imports.

Payload materialization is a metadata provider seam. Apple lazy-array lifetime,
bytes and model generations remain separate gates; these tests only verify the
actual project-owned path boundary and loader source ownership/cache behavior.
"""
import ast
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("owned_checkpoint_file_contract", ROOT / "src/tensorfold/cuda/tensor_file.py")
files = importlib.util.module_from_spec(spec)
spec.loader.exec_module(files)


def loader(family, provider):
    path = ROOT / "src/tensorfold/families" / family / "weights.py"
    source = ast.parse(path.read_bytes())
    cls = next(node for node in source.body if isinstance(node, ast.ClassDef) and node.name == "Weights")
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    namespace = {"Path": Path, "json": json, "mx": provider, "checkpoint_path": files.checkpoint_path,
                 "layouts": SimpleNamespace(detect=lambda index: "fixture", canonical=lambda name, layer: name),
                 "quant_formats": lambda config: (None, {})}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[future, cls], type_ignores=[])), str(path), "exec"), namespace)
    return namespace["Weights"]


class Paths(unittest.TestCase):
    def fixture(self, directory, where=None):
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=True)
        (root / "config.json").write_text("{}")
        (root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": where or {"a": "shard.safetensors", "b": "shard.safetensors"}}))
        return root

    def test_normal_nested_files_and_existing_one_shard_cache_unchanged(self):
        for family in ("glm5_next", "deepseek_v4"):
            with self.subTest(family=family), tempfile.TemporaryDirectory() as directory:
                root = self.fixture(directory, {"a": "nested/shard.safetensors", "b": "nested/shard.safetensors"})
                (root / "nested").mkdir()
                source = root / "nested/shard.safetensors"
                source.write_bytes(b"path-only fixture")
                calls = []
                one, two = object(), object()
                def load(path):
                    calls.append(path)
                    return {"a": one, "b": two}
                owner = loader(family, SimpleNamespace(load=load))(root)
                self.assertIs(owner.get("a"), one)
                self.assertIs(owner.get("b"), two)
                self.assertEqual(calls, [str(source.resolve())])

    def test_snapshot_can_borrow_only_its_own_blob_store(self):
        for family in ("glm5_next", "deepseek_v4"):
            with self.subTest(family=family), tempfile.TemporaryDirectory() as directory:
                base = Path(directory) / "models--org--model"
                root = self.fixture(base / "snapshots/revision")
                blob = base / "blobs/abc"
                blob.parent.mkdir()
                blob.write_bytes(b"path-only fixture")
                (root / "shard.safetensors").symlink_to(blob)
                calls = []
                def load(path):
                    calls.append(path)
                    return {"a": "value", "b": "other"}
                owner = loader(family, SimpleNamespace(load=load))(root)
                self.assertEqual(owner.get("a"), "value")
                self.assertEqual(calls, [str(blob.resolve())])

    def test_absolute_parent_traversal_and_external_link_refuse_before_any_payload(self):
        for family in ("glm5_next", "deepseek_v4"):
            for variant in ("parent", "absolute", "linked", "invalid_type", "empty"):
                with self.subTest(family=family, variant=variant), tempfile.TemporaryDirectory() as directory:
                    base = Path(directory)
                    root = base / "model"
                    outside = base / "outside.safetensors"
                    outside.write_bytes(b"unauthorized path-only fixture")
                    shard = {"parent": "../outside.safetensors", "absolute": str(outside),
                             "linked": "alias.safetensors", "invalid_type": [], "empty": ""}[variant]
                    self.fixture(root, {"a": shard})
                    if variant == "linked":
                        (root / shard).symlink_to(outside)
                    calls = []
                    def load(path):
                        calls.append(path)
                        return {}
                    with self.assertRaises(ValueError):
                        loader(family, SimpleNamespace(load=load))(root)
                    self.assertEqual(calls, [])

    def test_foreign_hf_blob_and_mutated_mapping_do_not_change_admitted_target(self):
        for family in ("glm5_next", "deepseek_v4"):
            with self.subTest(family=family), tempfile.TemporaryDirectory() as directory:
                base = Path(directory)
                root = self.fixture(base / "models--org--model/snapshots/revision", {"a": "alias.safetensors"})
                foreign = base / "models--other--repo/blobs/abc"
                foreign.parent.mkdir(parents=True)
                foreign.write_bytes(b"foreign path-only fixture")
                (root / "alias.safetensors").symlink_to(foreign)
                with self.assertRaises(ValueError):
                    loader(family, SimpleNamespace(load=lambda path: {}))(root)
        with tempfile.TemporaryDirectory() as directory:
            root = self.fixture(directory)
            (root / "shard.safetensors").write_bytes(b"path-only fixture")
            external = {"a": "shard.safetensors"}
            calls = []
            def load(path):
                calls.append(path)
                return {"a": "original"}
            owner = loader("deepseek_v4", SimpleNamespace(load=load))(root, external)
            external["a"] = "../foreign.safetensors"
            self.assertEqual(owner.get("a"), "original")
            self.assertEqual(calls, [str((root / "shard.safetensors").resolve())])

    def test_single_converted_file_checks_target_before_metadata_load(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            valid = root / "converted.safetensors"
            valid.write_bytes(b"path-only fixture")
            calls = []
            def load(path):
                calls.append(path)
                return {"a": "same"}
            kind = loader("deepseek_v4", SimpleNamespace(load=load))
            owner = kind.file(valid)
            self.assertEqual(owner.get("a"), "same")
            self.assertEqual(calls, [str(valid), str(valid)])
            foreign = root.parent / (root.name + "-foreign")
            # A symlink need not resolve to an existing target to be rejected.
            link = root / "linked.safetensors"
            link.symlink_to(foreign)
            before = list(calls)
            with self.assertRaises(ValueError):
                kind.file(link)
            self.assertEqual(calls, before)


if __name__ == "__main__":
    unittest.main()

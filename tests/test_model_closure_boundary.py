"""Real authorized loader tree membership/alias boundaries, standard library."""

import os
from pathlib import Path
import tempfile
import unittest

from tensorfold.engine.model_closure import LoaderClosure
from tensorfold.engine.model_closure import runtime_closure
from unittest.mock import patch
from importlib import metadata
from tensorfold.engine.model_identity import capture_model_identity
from snapshot_fd_transport import substitute_owners


class ClosureControls(unittest.TestCase):
    def setUp(self):
        substitute_owners(self)

    def test_runtime_captures_existing_bytecode_and_provider_native_files(self):
        with tempfile.TemporaryDirectory() as root:
            project = Path(root) / "project"
            project.mkdir()
            (project / "source.py").write_text("owned = 1\n")
            bytecode = project / "__pycache__"
            bytecode.mkdir()
            (bytecode / "source.pyc").write_bytes(b"independently execution-affecting bytecode")
            provider = Path(root) / "provider"
            (provider / "mlx").mkdir(parents=True)
            (provider / "mlx" / "core.so").write_bytes(b"owned native binary")

            class Installed:
                def locate_file(self, member):
                    return provider / member

            def select(name):
                if name == "mlx":
                    return Installed()
                raise metadata.PackageNotFoundError(name)

            with patch("tensorfold.engine.model_closure.metadata.distribution", side_effect=select):
                closure = runtime_closure(project)
            self.assertEqual(
                set(closure.files),
                {"tensorfold-source/source.py", "tensorfold-source/__pycache__/source.pyc", "provider-mlx/core.so"},
            )
            self.capture(closure)
            (bytecode / "source.pyc").write_bytes(b"different execution")
            with self.assertRaises(ValueError):
                closure.verify_unchanged()

    def fixture(self, root):
        model = Path(root) / "model"
        model.mkdir()
        (model / "config.json").write_text('{"model_type":"owned"}')
        (model / "tokenizer.json").write_text('{"vocab":{"a":1}}')
        (model / "weights.safetensors").write_bytes(b"owned checkpoint words")
        return model

    def capture(self, closure):
        return capture_model_identity(
            closure.files,
            runtime_identity={"data_schema": 1},
            authorize=closure.authorize,
            max_file_bytes=1 << 20,
            max_total_bytes=4 << 20,
        )

    def test_complete_model_and_drafter_data_hash_and_loaded_byte_change_refusal(self):
        with tempfile.TemporaryDirectory() as root:
            model = self.fixture(root)
            draft = Path(root) / "draft"
            draft.mkdir()
            (draft / "head.safetensors").write_bytes(b"selected head")
            closure = LoaderClosure({"model": model, "drafter": draft})
            self.assertEqual(
                set(closure.files),
                {"model/config.json", "model/tokenizer.json", "model/weights.safetensors", "drafter/head.safetensors"},
            )
            receipt = self.capture(closure)
            receipt.verify_unchanged()
            closure.verify_unchanged()
            (draft / "head.safetensors").write_bytes(b"changed head")
            with self.assertRaises(ValueError):
                receipt.verify_unchanged()
            with self.assertRaises(ValueError):
                closure.verify_unchanged()

    def test_hf_blob_files_and_internal_directory_aliases_admitted_without_cycles(self):
        with tempfile.TemporaryDirectory() as root:
            store = Path(root) / "models--owned"
            model = store / "snapshots" / "revision"
            model.mkdir(parents=True)
            blobs = store / "blobs"
            blobs.mkdir()
            (blobs / "hash").write_bytes(b"immutable HF data")
            (model / "model.safetensors").symlink_to(blobs / "hash")
            directory = model / "tokenizer"
            directory.mkdir()
            (directory / "vocab.json").write_text("{}")
            (model / "alias").symlink_to(directory, target_is_directory=True)
            closure = LoaderClosure({"model": model})
            self.assertEqual(
                set(closure.files), {"model/model.safetensors", "model/tokenizer/vocab.json", "model/alias/vocab.json"}
            )
            self.capture(closure)
            (directory / "cycle").symlink_to(model, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "cycle"):
                LoaderClosure({"model": model})

    def test_source_alias_retarget_is_rejected_even_when_original_target_is_unchanged(self):
        with tempfile.TemporaryDirectory() as root:
            model = self.fixture(root)
            first, second = model / "first", model / "second"
            first.write_bytes(b"first")
            second.write_bytes(b"second")
            alias = model / "selected.safetensors"
            alias.symlink_to(first)
            closure = LoaderClosure({"model": model})
            receipt = self.capture(closure)
            alias.unlink()
            alias.symlink_to(second)
            # Original resolved files can remain identical; complete alias
            # authority must still refuse changed loader selection.
            receipt.verify_unchanged()
            with self.assertRaises(ValueError):
                closure.verify_unchanged()
            with self.assertRaises(ValueError):
                closure.authorize("model/selected.safetensors", alias)

    def test_no_outside_tree_symlink_or_nonregular_input_is_admitted(self):
        with tempfile.TemporaryDirectory() as root:
            model = self.fixture(root)
            outside = Path(root) / "outside"
            outside.write_bytes(b"outside")
            alias = model / "alias"
            alias.symlink_to(outside)
            with self.assertRaises(ValueError):
                LoaderClosure({"model": model})
            alias.unlink()
            os.mkfifo(alias)
            with self.assertRaisesRegex(ValueError, "regular file"):
                LoaderClosure({"model": model})

    def test_entry_name_budgets_and_explicit_owned_external_sidecar(self):
        with tempfile.TemporaryDirectory() as root:
            model = self.fixture(root)
            sidecar = Path(root) / "selected-mtp.safetensors"
            sidecar.write_bytes(b"explicit current-loader sidecar")
            closure = LoaderClosure({"model": model}, extra_files={"mtp-head": sidecar})
            self.assertEqual(closure.authorize("sidecar/mtp-head", sidecar), sidecar.resolve())
            self.capture(closure)
            with self.assertRaises(ValueError):
                closure.authorize("sidecar/unselected", sidecar)
            with self.assertRaises(ValueError):
                LoaderClosure({"model": model}, max_files=2)
            with self.assertRaises(ValueError):
                LoaderClosure({"model": model}, max_name_bytes=4)
            (model / "empty").mkdir()
            with self.assertRaises(ValueError):
                LoaderClosure({"model": model}, max_files=3)

    def test_successor_generated_inputs_allowed_but_existing_input_mutation_never_is(self):
        with tempfile.TemporaryDirectory() as root:
            model = self.fixture(root)
            closure = LoaderClosure({"model": model})
            (model / "derived-head.safetensors").write_bytes(b"owned new conversion")
            with self.assertRaises(ValueError):
                closure.verify_unchanged()
            current = closure.verify_unchanged(allow_additions=True)
            self.assertEqual(len(current), 4)
            successor = LoaderClosure({"model": model})
            self.assertNotEqual(self.capture(closure).model_prefix, self.capture(successor).model_prefix)
            (model / "config.json").write_text('{"changed":true}')
            with self.assertRaises(ValueError):
                closure.verify_unchanged(allow_additions=True)


if __name__ == "__main__":
    unittest.main()

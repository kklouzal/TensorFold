"""Actual owned authorization and provider seam; no SDK or tensor execution."""

from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch
from snapshot_fd_transport import substitute_owners

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tensorfold.families.model_code import authorize_model_code, load_mlx_model  # noqa: E402 - direct stdlib script


class Boundary(unittest.TestCase):
    def setUp(self):
        substitute_owners(self)

    def test_interrupted_read_and_physically_closed_fd_preserve_malformed_note_primary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self.fixture(directory)
            primary, cleanup = KeyboardInterrupt("authorization read"), OSError("authorization close")
            primary.__notes__ = 123
            original = os.close
            closed = []

            def close(descriptor):
                original(descriptor)
                closed.append(descriptor)
                raise cleanup

            with (
                patch("tensorfold.families.model_code.os.read", side_effect=primary),
                patch("tensorfold.families.model_code.os.close", side_effect=close),
            ):
                with self.assertRaises(KeyboardInterrupt) as result:
                    authorize_model_code(root)
            self.assertIs(result.exception, primary)
            self.assertEqual(len(closed), 1)
            with self.assertRaises(OSError):
                os.fstat(closed[0])
            self.assertIsInstance(primary.__cause__, BaseExceptionGroup)
            self.assertIs(primary.__cause__.exceptions[0], cleanup)
            self.assertIsInstance(primary.__cause__.exceptions[1], TypeError)

    def fixture(self, directory, config=None):
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=True)
        (root / "config.json").write_text(json.dumps(config or {"model_type": "fixture"}))
        return root

    def provider(self, operation):
        provider = ModuleType("mlx_lm")
        provider.load = operation
        return patch.dict(sys.modules, {"mlx_lm": provider})

    def test_implicit_code_rejected_before_provider_and_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self.fixture(directory, {"model_file": "custom.py"})
            with self.provider(lambda *a, **k: self.fail("provider called")):
                with self.assertRaisesRegex(ValueError, "trust-model-code"):
                    load_mlx_model(root)
            self.assertFalse((root / "custom.py").exists())

    def test_supported_provider_override_blocks_reread_introducing_code(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self.fixture(directory)

            def load(path, **options):
                self.assertEqual(path, str(root.resolve()))
                self.assertEqual(
                    options, {"model_config": {"model_file": None}, "tokenizer_config": {"trust_remote_code": False}}
                )
                # Reproduce the documented provider merge at the executable seam.
                config = {"model_type": "fixture", "model_file": "malicious.py"}
                config.update(options["model_config"])
                self.assertIsNone(config["model_file"])
                return "model", "tokenizer"

            with self.provider(load):
                self.assertEqual(load_mlx_model(root), ("model", "tokenizer"))

    def test_explicit_custom_code_retained_and_paths_authorized(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self.fixture(directory, {"model_file": "nested/模型.py"})
            (root / "nested").mkdir()
            (root / "nested/模型.py").write_text("class Model: pass\n")
            proof = authorize_model_code(root, trust_model_code=True)
            self.assertTrue(proof.custom)
            calls = []
            with self.provider(lambda path, **options: calls.append(options) or (1, 2)):
                self.assertEqual(load_mlx_model(root, trust_model_code=True), (1, 2))
            self.assertEqual(
                calls,
                [{"model_config": {"model_file": "nested/模型.py"}, "tokenizer_config": {"trust_remote_code": True}}],
            )
            for name in ("../other.py", str(root / "nested/模型.py"), "", 0, False, "x\0.py"):
                self.fixture(directory, {"model_file": name})
                with self.assertRaises((ValueError, OSError)):
                    authorize_model_code(root, trust_model_code=True)

    def test_foreign_symlink_refused_and_own_hf_blob_supported(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            foreign = base / "foreign.py"
            foreign.write_text("pass\n")
            root = self.fixture(base / "plain", {"model_file": "custom.py"})
            (root / "custom.py").symlink_to(foreign)
            with self.assertRaisesRegex(ValueError, "authorized"):
                authorize_model_code(root, trust_model_code=True)
            store = base / "models--owned"
            blob = store / "blobs/code"
            blob.parent.mkdir(parents=True)
            blob.write_text("pass\n")
            root = self.fixture(store / "snapshots/rev", {"model_file": "custom.py"})
            (root / "custom.py").symlink_to(blob)
            self.assertEqual(authorize_model_code(root, trust_model_code=True).states[1][1], str(blob))

    def test_config_code_membership_and_bytes_mutation_fail_postflight(self):
        for name in ("config.json", "custom.py"):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                root = self.fixture(directory, {"model_file": "custom.py"})
                (root / "custom.py").write_text("pass\n")

                def load(*args, **kwargs):
                    if name.endswith("json"):
                        (root / name).write_text('{"model_file":"custom.py","changed":1}')
                    else:
                        (root / name).write_text("raise Exception()\n")
                    return 1, 2

                with self.provider(load), self.assertRaisesRegex(RuntimeError, "changed"):
                    load_mlx_model(root, trust_model_code=True)

    def test_primary_loader_failure_preserved_through_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self.fixture(directory)
            primary = KeyboardInterrupt("loader interrupted")

            def load(*args, **kwargs):
                (root / "config.json").write_text('{"changed":1}')
                raise primary

            with self.provider(load), self.assertRaises(KeyboardInterrupt) as caught:
                load_mlx_model(root)
            self.assertIs(caught.exception, primary)
            self.assertIsInstance(primary.__cause__, RuntimeError)
            self.assertTrue(primary.__notes__)

    def test_post_loader_validation_malformed_notes_preserves_primary_and_secondary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self.fixture(directory)
            primary = KeyboardInterrupt("loader interrupted")
            primary.__notes__ = 123

            def load(*args, **kwargs):
                (root / "config.json").write_text('{"changed":1}')
                raise primary

            with self.provider(load), self.assertRaises(KeyboardInterrupt) as caught:
                load_mlx_model(root)
            self.assertIs(caught.exception, primary)
            self.assertIsInstance(primary.__cause__, BaseExceptionGroup)
            self.assertIsInstance(primary.__cause__.exceptions[0], RuntimeError)
            self.assertIsInstance(primary.__cause__.exceptions[1], TypeError)

    def test_same_primary_validation_and_descriptor_close_never_self_chain(self):
        from tensorfold.families.model_code import ModelCode

        with tempfile.TemporaryDirectory() as directory:
            root = self.fixture(directory)
            primary = KeyboardInterrupt("same failure")
            with (
                self.provider(lambda *args, **kwargs: (_ for _ in ()).throw(primary)),
                patch.object(ModelCode, "validate", side_effect=primary),
                self.assertRaises(KeyboardInterrupt) as caught,
            ):
                load_mlx_model(root)
            self.assertIs(caught.exception, primary)
            self.assertIsNot(primary.__cause__, primary)
            original = os.close

            def close(fd):
                original(fd)
                raise primary

            with (
                patch("tensorfold.families.model_code.os.read", side_effect=primary),
                patch("tensorfold.families.model_code.os.close", side_effect=close),
                self.assertRaises(KeyboardInterrupt) as caught,
            ):
                authorize_model_code(root)
            self.assertIs(caught.exception, primary)
            self.assertIsNot(primary.__cause__, primary)

    def test_fifo_and_unknown_config_override_refused_without_blocking(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self.fixture(directory, {"model_file": "pipe"})
            os.mkfifo(root / "pipe")
            with self.assertRaisesRegex(ValueError, "regular"):
                authorize_model_code(root, trust_model_code=True)
            self.fixture(directory)
            with (
                self.provider(lambda *a, **k: self.fail("provider called")),
                self.assertRaisesRegex(ValueError, "overrides"),
            ):
                load_mlx_model(root, model_config={"model_file": "other.py"})

    def test_owned_tokenizer_always_forwards_no_code_including_trimmed_retry(self):
        source = ast.parse((ROOT / "src/tensorfold/families/tokenizer.py").read_text())
        node = next(n for n in source.body if isinstance(n, ast.FunctionDef) and n.name == "load_tokenizer")
        provider = ModuleType("mlx_lm.utils")
        for retry in (False, True):
            calls = []
            config = object()

            def load(path, **options):
                calls.append(options)
                self.assertIs(options["tokenizer_config_extra"]["trust_remote_code"], False)
                self.assertEqual(options["eos_token_ids"], [1, 2])
                if retry and len(calls) == 1:
                    raise ValueError("layer_types contains MTP entries")
                return "tokenizer"

            provider.load_tokenizer = load
            namespace = {"Path": Path, "Any": object, "trimmed_config": lambda _: config}
            module = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
            exec(compile(module, "actual_owned_tokenizer", "exec"), namespace)
            with patch.dict(sys.modules, {"mlx_lm.utils": provider}):
                self.assertEqual(namespace["load_tokenizer"](Path("fixture"), [1, 2]), "tokenizer")
            self.assertEqual(len(calls), 2 if retry else 1)
            if retry:
                self.assertIs(calls[-1]["tokenizer_config_extra"]["config"], config)

    def test_per_process_reuse_identity_retains_bounded_stable_eviction_owner(self):
        source = ast.parse((ROOT / "src/tensorfold/engine/prefix_snapshots.py").read_text())
        nodes = [n for n in source.body if isinstance(n, ast.FunctionDef) and n.name in ("model_group", "snapshot_key")]
        namespace = {"hashlib": hashlib, "Sequence": list}
        exec(
            compile(
                ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), "actual_snapshot_identity", "exec"
            ),
            namespace,
        )
        root = "a" * 64
        first, second = f"custom-code-v1:{root}:{'1' * 32}|mlx=1", f"custom-code-v1:{root}:{'2' * 32}|mlx=1"
        group, key = namespace["model_group"], namespace["snapshot_key"]
        self.assertEqual(group(first), group(second))
        self.assertNotEqual(key(first, [1, 2]), key(second, [1, 2]))
        self.assertNotEqual(first.split("|")[0], second.split("|")[0])  # no old-token warming either
        self.assertNotEqual(group(first), group(f"custom-code-v1:{'b' * 64}:{'1' * 32}"))
        for malformed in (f"/models/custom-code-v1:{root}:{'1' * 32}", f"custom-code-v1:{root}:short", "/model|mlx=1"):
            self.assertEqual(group(malformed), malformed.split("|")[0])
        pruning = ast.parse((ROOT / "src/tensorfold/server/checkpoints.py").read_text())
        prune = next(n for n in pruning.body if isinstance(n, ast.FunctionDef) and n.name == "prune_conversations")
        self.assertIn("model_group", ast.unparse(prune))

    def test_cli_opt_in_refuses_backends_and_recipes_that_cannot_honor_it(self):
        from tensorfold.cli import build_parser
        from tensorfold.serve_options import check

        args = build_parser().parse_args(["serve", "fixture", "--trust-model-code", "--no-drafts"])
        self.assertTrue(args.trust_model_code)
        for backend, supported in (("cuda", True), ("mlx", False)):
            family = SimpleNamespace(title="fixture", package=SimpleNamespace(MLX_MODEL_FILE=supported))
            with self.assertRaisesRegex(ValueError, "trust-model-code"):
                check(args, family, backend)
        family = SimpleNamespace(title="fixture", package=SimpleNamespace(MLX_MODEL_FILE=True))
        check(args, family, "mlx")
        for name in ("qwen3_5", "qwen3_5_moe", "gemma4", "nemotron_h"):
            tree = ast.parse((ROOT / "src/tensorfold/families" / name / "__init__.py").read_text())
            load = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "load")
            self.assertIn("trust_model_code", [n.arg for n in load.args.kwonlyargs])
            self.assertTrue(any(isinstance(n, ast.keyword) and n.arg == "trust_model_code" for n in ast.walk(load)))


if __name__ == "__main__":
    unittest.main()

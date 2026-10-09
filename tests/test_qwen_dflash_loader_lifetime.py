"""Current DFlash reader/config boundary controls without importing any SDK.

Execute the exact constructor through its owned file-read boundary, then stop
before native numeric setup. Original arithmetic/projection/packing AST proof
is retained with the source handoff; these controls qualify owner transport.
"""

import ast
import __future__
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

REPO = Path(__file__).resolve().parents[1]
SOURCE = REPO / "src/tensorfold/families/qwen3_5/cuda/dflash2.py"
SPEC = importlib.util.spec_from_file_location("owned_dflash_tensor_file", REPO / "src/tensorfold/cuda/tensor_file.py")
FILES = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FILES)


class BoundaryDone(Exception):
    pass


class Controls(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "model"
        self.root.mkdir()
        self.config = {
            "hidden_size": 64,
            "head_dim": 64,
            "num_attention_heads": 8,
            "num_key_value_heads": 2,
            "rms_norm_eps": 1e-6,
            "rope_parameters": {"rope_theta": 10000},
            "dflash_config": {"mask_token_id": 1, "block_size": 8, "conv_group_size": 64},
            "num_hidden_layers": 1,
            "sliding_window": 2048,
        }
        (self.root / "config.json").write_text(json.dumps(self.config))
        (self.root / "model.safetensors").write_bytes(b"owned reader fixture")
        self.closed = self.acquired = 0
        self.error = self.close_error = None
        self.fail = ""
        self.reads = []
        outer = self

        class Value:
            def float(self):
                if outer.fail == "convert":
                    raise outer.error
                return self

            def numpy(self):
                return self

            def copy(self):
                return self

        self.value = Value()

        class Reader:
            def __init__(self, paths):
                outer.acquired += 1
                outer.paths = paths

            def keys(self):
                if outer.fail == "keys":
                    raise outer.error
                return ["candidate_selector.predecessor_codebook", "layers.0.weight"]

            def get(self, name):
                outer.reads.append(name)
                if outer.fail == "get":
                    raise outer.error
                return outer.value

            def close(self):
                outer.closed += 1
                if outer.close_error is not None:
                    raise outer.close_error

        tree = ast.parse(SOURCE.read_bytes())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "DFlash2")
        init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
        last = next(
            i
            for i, n in enumerate(init.body)
            if isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Attribute) and t.attr == "inv_freq" for t in n.targets)
        )
        init.body = init.body[:last] + [
            ast.Raise(exc=ast.Call(func=ast.Name(id="BoundaryDone", ctx=ast.Load()), args=[], keywords=[]), cause=None)
        ]
        ns = {
            "Path": Path,
            "read_metadata_json": FILES.read_metadata_json,
            "checkpoint_path": FILES.checkpoint_path,
            "SafeTensors": Reader,
            "BoundaryDone": BoundaryDone,
        }
        exec(
            compile(
                ast.fix_missing_locations(ast.Module([init], [])),
                str(SOURCE),
                "exec",
                flags=__future__.annotations.compiler_flag,
            ),
            ns,
        )
        self.init = ns["__init__"]
        self.target = SimpleNamespace(embed=object(), norm=SimpleNamespace(device="opaque owner"))
        self.owner = SimpleNamespace()

    def invoke(self, path=None):
        return self.init(self.owner, self.root if path is None else path, self.target)

    def test_complete_reads_close_once_before_numeric_setup(self):
        with self.assertRaises(BoundaryDone):
            self.invoke()
        self.assertEqual((self.acquired, self.closed), (1, 1))
        self.assertEqual(self.reads, ["candidate_selector.predecessor_codebook", "layers.0.weight"])
        self.assertIs(self.owner.weights["layers.0.weight"], self.value)

    def test_keys_read_and_conversion_interruption_close_once(self):
        for point in ("keys", "get", "convert"):
            self.fail = point
            self.error = KeyboardInterrupt(point)
            self.acquired = self.closed = 0
            with self.subTest(point=point), self.assertRaises(KeyboardInterrupt) as caught:
                self.invoke()
            self.assertIs(caught.exception, self.error)
            self.assertEqual((self.acquired, self.closed), (1, 1))

    def test_close_failure_preserves_original_primary_and_once_only(self):
        self.fail = "get"
        self.error = KeyboardInterrupt("read")
        self.close_error = OSError("drain")
        with self.assertRaises(KeyboardInterrupt) as caught:
            self.invoke()
        self.assertIs(caught.exception, self.error)
        self.assertIs(caught.exception.__cause__, self.close_error)
        self.assertEqual(self.closed, 1)
        self.assertTrue(any("cleanup" in n for n in caught.exception.__notes__))

    def test_success_close_failure_is_visible(self):
        self.close_error = OSError("close")
        with self.assertRaises(OSError) as caught:
            self.invoke()
        self.assertIs(caught.exception, self.close_error)
        self.assertEqual(self.closed, 1)

    def test_config_duplicates_nonfinite_and_nonobject_refuse_before_reader(self):
        for content in ('{"hidden_size":64,"hidden_size":32}', '{"eps":NaN}', '{"eps":1e400}', "[]"):
            (self.root / "config.json").write_text(content)
            with self.subTest(content=content), self.assertRaises(ValueError):
                self.invoke()
            self.assertEqual(self.acquired, 0)

    def test_foreign_symlink_refuses_before_reader(self):
        outside = Path(self.temp.name) / "foreign.json"
        outside.write_text(json.dumps(self.config))
        (self.root / "config.json").unlink()
        (self.root / "config.json").symlink_to(outside)
        with self.assertRaises(ValueError):
            self.invoke()
        self.assertEqual(self.acquired, 0)

    def test_hf_config_and_shard_blob_links_remain_authorized(self):
        model = Path(self.temp.name) / "models--owner--model"
        blobs = model / "blobs"
        blobs.mkdir(parents=True)
        snapshot = model / "snapshots" / "revision"
        snapshot.mkdir(parents=True)
        (blobs / "config").write_text(json.dumps(self.config))
        (blobs / "tensor").write_bytes(b"owned blob fixture")
        (snapshot / "config.json").symlink_to("../../blobs/config")
        (snapshot / "model.safetensors").symlink_to("../../blobs/tensor")
        with self.assertRaises(BoundaryDone):
            self.invoke(snapshot)
        self.assertEqual(self.paths, [blobs / "tensor"])
        self.assertEqual(self.closed, 1)


if __name__ == "__main__":
    unittest.main()

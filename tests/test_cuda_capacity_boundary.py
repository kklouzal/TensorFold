"""Strict checkpoint preflight with real isolated files and no numeric runtime."""
import json
import os
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch

import tensorfold.cuda.tensor_file as metadata_contract

from tensorfold.cuda import capacity
from tensorfold.cuda.tensor_file import checkpoint_path, read_metadata_json, read_header


def write_tensor(path, entries, payload=b""):
    raw = entries if isinstance(entries, bytes) else json.dumps(entries).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + payload)


def entry(dtype="U8", shape=None, offsets=None):
    return {"dtype": dtype, "shape": [4] if shape is None else shape,
            "data_offsets": [0, 4] if offsets is None else offsets}


class CapacityBoundary(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_valid_scalar_empty_metadata_and_header_shapes_are_retained(self):
        tensors = {"empty": entry(shape=[0, 9], offsets=[0, 0]),
                   "scalar": entry("F32", [], [0, 4]), "__metadata__": {"owner": "α"}}
        write_tensor(self.root / "model.safetensors", tensors, b"1234")
        got = capacity.headers(self.root)
        self.assertEqual(set(got), {"empty", "scalar"})
        self.assertEqual(got["empty"]["shape"], [0, 9])
        self.assertEqual(got["scalar"]["shape"], [])
        self.assertFalse(got["scalar"]["split"])

    @unittest.skipUnless(hasattr(os, "mkfifo"), "POSIX FIFO fixture")
    def test_metadata_and_tensor_fifo_refuse_without_waiting_for_a_writer(self):
        path = self.root / "checkpoint-fifo"
        os.mkfifo(path)
        for operation in (lambda: read_metadata_json(path), lambda: read_header(path, {"U8": 1})):
            with self.assertRaisesRegex(ValueError, "regular file"):
                operation()

    @unittest.skipUnless(os.name == "posix", "POSIX device fixture")
    def test_opened_nonregular_device_is_refused(self):
        with self.assertRaisesRegex(ValueError, "regular file"):
            read_metadata_json("/dev/null")

    def test_zero_open_flag_retains_ordinary_file_and_hf_link_reads(self):
        path = self.root / "metadata.json"
        path.write_text('{"owner":"α"}')
        linked = self.root / "linked.json"
        linked.symlink_to(path.name)
        with patch.object(metadata_contract, "_OPEN_NONBLOCK", 0):
            self.assertEqual(read_metadata_json(path), {"owner": "α"})
            self.assertEqual(read_metadata_json(linked), {"owner": "α"})

    def test_header_refuses_truncation_overlap_gap_wrong_types_and_duplicate_json(self):
        p = self.root / "model.safetensors"
        invalid = [(b"{\"x\":1,\"x\":2}", b""),
                   ({"a": entry(shape=[True])}, b"1234"),
                   ({"a": entry(), "b": entry()}, b"1234"),
                   ({"a": entry(offsets=[1, 5])}, b"12345"),
                   ({"a": entry()}, b""), ({"a": entry()}, b"12345"),
                   ({"a": entry("U8", [], [0, 4])}, b"1234")]
        for metadata, data in invalid:
            with self.subTest(metadata=metadata):
                write_tensor(p, metadata, data)
                with self.assertRaises((ValueError, OSError)):
                    capacity.headers(self.root)
        p.write_bytes(b"abc")
        with self.assertRaisesRegex(ValueError, "truncated"):
            capacity.headers(self.root)

    def test_index_authorizes_all_files_before_reading_them(self):
        outside = self.root.parent / (self.root.name + "-outside.safetensors")
        self.addCleanup(lambda: outside.unlink(missing_ok=True))
        write_tensor(outside, {"x": entry()}, b"1234")
        (self.root / "escape.safetensors").symlink_to(outside)
        index = self.root / "model.safetensors.index.json"
        for name in (str(outside), "../" + outside.name, "escape.safetensors"):
            index.write_text(json.dumps({"weight_map": {"x": name}}))
            with self.assertRaisesRegex(ValueError, "absolute|traverse|authorized|directory"):
                capacity.headers(self.root)
        with self.assertRaisesRegex(ValueError, "directory"):
            capacity.headers(self.root, files=[outside])

    def test_relative_model_accepts_absolute_same_root_explicit_files(self):
        p = self.root / "model.safetensors"
        write_tensor(p, {"x": entry()}, b"1234")
        previous = Path.cwd()
        try:
            os.chdir(self.root.parent)
            got = capacity.headers(Path(self.root.name), files=[p])
            self.assertEqual(set(got), {"x"})
        finally:
            os.chdir(previous)

    def test_all_index_target_authorization_precedes_first_payload_read(self):
        (self.root / "a.safetensors").write_bytes(b"invalid first header")
        (self.root / "model.safetensors.index.json").write_text(json.dumps(
            {"weight_map": {"a": "a.safetensors", "b": "../outside.safetensors"}}))
        with self.assertRaisesRegex(ValueError, "traverse"):
            capacity.headers(self.root)

    def test_symlinked_hf_blob_root_cannot_expand_its_authority(self):
        store = self.root / "models--owner--model"
        root = store / "snapshots" / "revision"
        root.mkdir(parents=True)
        external = self.root / "unrelated"
        external.mkdir()
        (store / "blobs").symlink_to(external, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "outside its own store"):
            checkpoint_path(root, "model.safetensors")

    def test_standard_hf_snapshot_blob_links_are_shared_by_both_boundaries(self):
        store = self.root / "models--owner--model"
        root = store / "snapshots" / "revision"
        blobs = store / "blobs"
        root.mkdir(parents=True)
        blobs.mkdir()
        write_tensor(blobs / "payload", {"x": entry()}, b"1234")
        (root / "model.safetensors").symlink_to("../../blobs/payload")
        (root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"x": "model.safetensors"}}))
        self.assertEqual(checkpoint_path(root, "model.safetensors"), blobs / "payload")
        self.assertEqual(set(capacity.headers(root)), {"x"})

    def test_index_schema_duplicates_missing_and_wrong_tensor_shards(self):
        write_tensor(self.root / "a.safetensors", {"a": entry()}, b"1234")
        write_tensor(self.root / "b.safetensors", {"b": entry()}, b"1234")
        p = self.root / "model.safetensors.index.json"
        cases = ['{"weight_map":{"a":"a.safetensors","a":"b.safetensors"}}',
                 json.dumps({"weight_map": {"a": "b.safetensors", "b": "a.safetensors"}}),
                 json.dumps({"weight_map": {"missing": "a.safetensors"}}),
                 json.dumps({"weight_map": {"a": ["a.safetensors"]}}),
                 json.dumps({"weight_map": {}}), json.dumps([])]
        for raw in cases:
            p.write_text(raw)
            with self.assertRaises(ValueError):
                capacity.headers(self.root)
        p.write_text(json.dumps({"weight_map": {"a": "a.safetensors", "b": "b.safetensors"}}))
        self.assertEqual(set(capacity.headers(self.root)), {"a", "b"})

    def test_rank_split_contract_and_invalid_rank(self):
        write_tensor(self.root / "model.rank0.safetensors", {"x": entry()}, b"1234")
        self.assertTrue(capacity.headers(self.root, rank=0)["x"]["split"])
        for rank in (True, -1, 0.0, "0"):
            with self.assertRaises(ValueError):
                capacity.headers(self.root, rank=rank)
        with self.assertRaisesRegex(ValueError, "another rank"):
            capacity.headers(self.root, rank=1)

    def test_config_selection_keeps_valid_existing_precedence(self):
        p = self.root / "config.json"
        p.write_text(json.dumps({"hidden_size": 9, "text_config": {"hidden_size": 4},
                                "quantization": {"bits": 4}, "quantization_config": {"bits": 8}}))
        self.assertEqual(capacity.config(self.root), {"hidden_size": 4, "_quantization": {"bits": 4}})
        p.write_text(json.dumps({"hidden_size": 9, "text_config": None}))
        self.assertEqual(capacity.config(self.root)["hidden_size"], 9)

    def test_metadata_refuses_invalid_json_encoding_numbers_and_shape(self):
        p = self.root / "config.json"
        cases = [b'{"x":1,"x":2}', b'{"x":NaN}', b'{"x":1e999}',
                 b'{"x":"\\ud800"}', b'{"text_config":[]}', b'{"quantization":[1]}',
                 b'{"quantization":[]}', b'{"quantization_config":false}',
                 b'[]', b'{}\xff', '{"x":1}'.encode("utf-16")]
        for data in cases:
            p.write_bytes(data)
            with self.assertRaises(ValueError):
                capacity.config(self.root)
        depth = sys.getrecursionlimit() + 100
        p.write_bytes(b'{"x":' + b'[' * depth + b'0' + b']' * depth + b'}')
        try:
            parsed = read_metadata_json(p)["x"]
        except ValueError as error:
            self.assertIn("nesting", str(error))
        else:
            # JSON parsers with iterative nesting support retain the data.
            for _ in range(depth):
                parsed = parsed[0]
            self.assertEqual(parsed, 0)


if __name__ == "__main__":
    unittest.main()

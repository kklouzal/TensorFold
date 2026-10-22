"""Execute real SafeTensors constructor/close methods with stdlib reader owners.

Opaque headers qualify mapping and failure ownership, without importing Torch
or exercising numerical/native IO. Real direct-read gates remain separate.
"""

import ast
import __future__
from pathlib import Path
import unittest

SOURCE = Path(__file__).resolve().parents[1] / "src/tensorfold/cuda/direct_read.py"


class Controls(unittest.TestCase):
    def setUp(self):
        outer = self
        self.created = []
        self.error = self.close_error = None
        self.at = 0
        self.headers = None

        class Reader:
            def __init__(self):
                self.closed = 0
                outer.created.append(self)

            def close(self):
                self.closed += 1
                if outer.close_error is not None:
                    raise outer.close_error

        self.Reader = Reader

        def header(path):
            outer.at += 1
            if outer.error is not None and outer.at == 2:
                raise outer.error
            if outer.headers is not None:
                return outer.headers[path]
            return 32, {"__metadata__": {}, "value": {"data_offsets": [0, 4], "dtype": "U8", "shape": [4]}}

        tree = ast.parse(SOURCE.read_bytes())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "SafeTensors")
        ns = {"Reader": Reader, "Path": Path, "read_header": header}
        exec(
            compile(
                ast.fix_missing_locations(ast.Module([cls], [])),
                str(SOURCE),
                "exec",
                flags=__future__.annotations.compiler_flag,
            ),
            ns,
        )
        self.Files = ns["SafeTensors"]
        self.namespace = ns

    def test_one_shard_shares_lexical_path_but_not_shape_lists(self):
        shape = [2]
        self.headers = {"shard/../weights": (32, {
            "first": {"data_offsets": [0, 2], "dtype": "U8", "shape": shape},
            "__metadata__": {},
            "second": {"data_offsets": [2, 4], "dtype": "U8", "shape": shape},
        })}
        files = self.Files(["shard/../weights"])
        first, second = files.where["first"], files.where["second"]
        self.assertIs(first[0], second[0])
        self.assertEqual(first[0], Path("shard/../weights"))
        self.assertIn("..", first[0].parts)
        self.assertEqual((first[1:4], second[1:4]), ((32, 2, "U8"), (34, 2, "U8")))
        self.assertIsNot(first[4], second[4])
        first[4][0] = 9
        self.assertEqual(second[4], [2])
        self.assertEqual(shape, [2])
        files.close()
        self.assertEqual(files.reader.closed, 1)

    def test_shard_local_paths_preserve_later_duplicate_winner_and_key_order(self):
        entry = {"data_offsets": [0, 1], "dtype": "U8", "shape": [1]}
        self.headers = {"first": (32, {"duplicate": entry, "retained": entry}),
                        "second": (64, {"duplicate": entry, "last": entry})}
        files = self.Files(["first", "second"])
        self.assertEqual(files.keys(), ["duplicate", "retained", "last"])
        self.assertEqual(files.where["duplicate"], (Path("second"), 64, 1, "U8", [1]))
        self.assertIs(files.where["duplicate"][0], files.where["last"][0])
        self.assertEqual(files.where["retained"][0], Path("first"))
        self.assertIsNot(files.where["retained"][0], files.where["duplicate"][0])
        files.close()

    def test_empty_and_metadata_only_shards_do_not_construct_mapping_paths(self):
        self.headers = {"empty": (8, {}), "metadata": (32, {"__metadata__": {}})}

        def unexpected(_path):
            raise AssertionError("a shard without tensor records must not allocate a mapping path")

        self.namespace["Path"] = unexpected
        files = self.Files(["empty", "metadata"])
        self.assertEqual(files.keys(), [])
        self.assertEqual(self.at, 2)
        self.assertEqual(files.reader.closed, 0)
        files.close()
        self.assertEqual(files.reader.closed, 1)

    def test_mapping_path_failure_keeps_reader_ownership_and_original_exception(self):
        failure = MemoryError("mapping path")
        cause = ValueError("prior cause")
        failure.__cause__ = cause

        def unavailable(_path):
            raise failure

        self.namespace["Path"] = unavailable
        with self.assertRaises(MemoryError) as caught:
            self.Files(["first"])
        self.assertIs(caught.exception, failure)
        self.assertIs(caught.exception.__cause__, cause)
        self.assertEqual(self.created[-1].closed, 1)
        borrowed = self.Reader()
        with self.assertRaises(MemoryError) as caught:
            self.Files(["first"], borrowed)
        self.assertIs(caught.exception, failure)
        self.assertEqual(borrowed.closed, 0)
        borrowed.close()

    def test_header_failure_precedes_any_mapping_path_construction(self):
        self.headers = {"first": (8, {})}
        failure = self.error = ValueError("second header")

        def unexpected(_path):
            raise AssertionError("header failure must remain the primary")

        self.namespace["Path"] = unexpected
        with self.assertRaises(ValueError) as caught:
            self.Files(["first", "second"])
        self.assertIs(caught.exception, failure)
        self.assertEqual(self.created[-1].closed, 1)

    def test_original_success_mapping_later_file_wins_and_close_is_explicit(self):
        files = self.Files(["first", "second"])
        self.assertEqual(files.keys(), ["value"])
        self.assertEqual(files.where["value"], (Path("second"), 32, 4, "U8", [4]))
        self.assertEqual(files.reader.closed, 0)
        files.close()
        self.assertEqual(files.reader.closed, 1)

    def test_owned_header_failure_and_interruption_close_once(self):
        for error in (ValueError("header"), KeyboardInterrupt("header")):
            self.at = 0
            self.error = error
            with self.subTest(error=type(error).__name__), self.assertRaises(type(error)) as caught:
                self.Files(["first", "second"])
            self.assertIs(caught.exception, error)
            self.assertEqual(self.created[-1].closed, 1)

    def test_owned_close_failure_preserves_header_primary(self):
        self.error = KeyboardInterrupt("header")
        self.close_error = OSError("drain")
        with self.assertRaises(KeyboardInterrupt) as caught:
            self.Files(["first", "second"])
        self.assertIs(caught.exception, self.error)
        self.assertIs(caught.exception.__cause__, self.close_error)
        self.assertEqual(self.created[-1].closed, 1)
        self.assertTrue(any("cleanup" in n for n in caught.exception.__notes__))

    def test_borrowed_reader_is_untouched_after_constructor_failure(self):
        borrowed = self.Reader()
        self.error = ValueError("header")
        with self.assertRaises(ValueError) as caught:
            self.Files(["first", "second"], borrowed)
        self.assertIs(caught.exception, self.error)
        self.assertEqual(borrowed.closed, 0)
        self.assertEqual(len(self.created), 1)

    def test_existing_explicit_close_on_supplied_reader_is_preserved(self):
        borrowed = self.Reader()
        files = self.Files(["first"], borrowed)
        self.assertIs(files.reader, borrowed)
        files.close()
        self.assertEqual(borrowed.closed, 1)

    def test_existing_false_supplied_reader_selection_and_failure_owner(self):
        class FalseReader(self.Reader):
            def __bool__(self):
                return False

        borrowed = FalseReader()
        self.error = ValueError("header")
        with self.assertRaises(ValueError):
            self.Files(["first", "second"], borrowed)
        self.assertEqual(borrowed.closed, 0)
        self.assertEqual(len(self.created), 2)
        self.assertEqual(self.created[-1].closed, 1)


if __name__ == "__main__":
    unittest.main()

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

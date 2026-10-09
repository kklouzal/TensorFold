"""Draft-ID source controls with stdlib files and an allocation-recording stub."""
import ast
import bz2
import codecs
from contextlib import contextmanager
import gzip
import os
from pathlib import Path
import stat
import tempfile
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'src/tensorfold/families/qwen4_exp/cuda/weight_types.py'


def api(*, namespace=False):
    tree = ast.parse(SOURCE.read_text())
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                and node.name in ('draft_token_ids', '_draft_file_ids', '_draft_id_stream')]
    calls = []

    def arange(count, **kwargs):
        calls.append(('arange', count))
        return list(range(count))

    def fromiter(ids, **kwargs):
        result = list(ids)
        calls.append(('fromiter', kwargs['count']))
        return result

    ns = {'Path': Path, '__file__': str(SOURCE), 'contextmanager': contextmanager,
          'os': os, 'stat': stat, 'gzip': gzip, 'bz2': bz2, 'codecs': codecs,
          'np': SimpleNamespace(int64=object(), arange=arange, fromiter=fromiter)}
    future = ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[future, *selected], type_ignores=[])),
                 str(SOURCE), 'exec'), ns)
    return (ns if namespace else ns['draft_token_ids']), calls


class DraftIds(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / 'ids.txt'
        self.ids, self.calls = api()

    def test_none_zero_and_bounded_numeric_count(self):
        self.assertIsNone(self.ids(None, 8))
        self.assertIsNone(self.ids(0, 8))
        self.assertEqual(self.ids(3, 8), [0, 1, 2])
        self.assertEqual(self.ids(10**100, 8), list(range(8)))
        self.assertEqual(self.calls, [('arange', 3), ('arange', 8)])

    def test_exact_sorted_ids_multicolumn_comments_duplicates_and_upper_filter(self):
        self.path.write_text('7 2 2 # ignored -99 2.2\r\n+0 -0 0003\n20 8 # upper IDs filtered\n')
        self.assertEqual(self.ids(str(self.path), 8), [0, 2, 3, 7])
        self.assertEqual(self.calls, [('fromiter', 4)])

    def test_empty_or_invalid_file_refuses_before_array_allocation(self):
        for value in ('', '# only comments', '8 90', '-1 2', '2.1', '1e-2', 'NaN', '1+2', '+',
                      str(2**63), '2e+', '2e', '2..0', '2.3e0', '-.1e1', '1e9999999999999'):
            self.path.write_text(value)
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.ids(str(self.path), 8)
        self.assertEqual(self.calls, [])

    def test_invalid_config_and_count_types_refuse_before_array(self):
        for value in (-1, True, False, 1.5, [], ''):
            with self.assertRaises((ValueError, TypeError)):
                self.ids(value, 8)
        for vocab in (0, -1, True, 1.5, 2**63):
            with self.assertRaises(ValueError):
                self.ids(4, vocab)
        self.assertEqual(self.calls, [])

    def test_chunk_boundaries_long_zero_tokens_and_comments(self):
        self.path.write_bytes(b'#' + b'-99 ignored ' * 20000 + b'\n' + b'0' * 200000 + b'7\n+2\n')
        self.assertEqual(self.ids(str(self.path), 8), [2, 7])
        self.path.write_bytes(b' ' * ((64 << 10) - 1) + b'-1\n')
        with self.assertRaises(ValueError):
            self.ids(str(self.path), 8)

    def test_exact_integral_decimal_scientific_and_unicode_whitespace(self):
        self.path.write_text('+2.0 3e0\N{NO-BREAK SPACE}.4e1\n50e-1 6000e-3 -0e-999999999999 0.7E+1\n')
        self.assertEqual(self.ids(str(self.path), 8), [0, 2, 3, 4, 5, 6, 7])
        self.path.write_text('922337203685477580700000e-5 7\n')
        self.assertEqual(self.ids(str(self.path), 8), [7])
        self.path.write_text('0.' + '0' * 200000 + '2e200001\n')
        self.assertEqual(self.ids(str(self.path), 8), [2])

    def test_compressed_expansion_is_chunked_and_matches_plain(self):
        raw = b'#' + b'ignored ' * 100000 + b'\n7.0 2e0 2.0\n'
        for suffix, encode in (('.gz', gzip.compress), ('.bz2', bz2.compress)):
            path = self.path.with_suffix(suffix)
            path.write_bytes(encode(raw))
            self.assertEqual(self.ids(str(path), 8), [2, 7])
            path.write_bytes(b'invalid compressed input')
            with self.assertRaises((ValueError, OSError, EOFError)):
                self.ids(str(path), 8)

    @unittest.skipUnless(hasattr(os, 'mkfifo'), 'FIFO filesystem contract')
    def test_fifo_refuses_without_waiting_for_writer(self):
        os.mkfifo(self.path)
        with self.assertRaisesRegex(ValueError, 'regular file'):
            self.ids(str(self.path), 8)
        self.assertEqual(self.calls, [])

    def test_cleanup_attempts_all_owners_and_preserves_parser_primary(self):
        namespace, _ = api(namespace=True)
        path = self.path.with_suffix('.gz')
        path.write_bytes(b'owned fixture')
        raw = path.open('rb')
        closed = []

        class Raw:
            def fileno(self):
                return raw.fileno()

            def close(self):
                raw.close()
                closed.append('raw')
                raise OSError('raw close failed')

        class Decoder:
            def close(self):
                closed.append('decoder')
                raise OSError('decoder close failed')

        namespace['open'] = lambda *args, **kwargs: Raw()
        namespace['gzip'] = SimpleNamespace(GzipFile=lambda **kwargs: Decoder())
        primary = ValueError('parser primary')
        with self.assertRaises(ValueError) as caught:
            with namespace['_draft_id_stream'](path):
                raise primary
        self.assertIs(caught.exception, primary)
        self.assertEqual(closed, ['decoder', 'raw'])
        self.assertTrue(raw.closed)
        self.assertEqual(len(primary.__notes__), 2)

    def test_utf8_and_compression_failures_allocate_no_array(self):
        for suffix, encode in (('.txt', lambda data: data), ('.gz', gzip.compress), ('.bz2', bz2.compress)):
            path = self.path.with_suffix(suffix)
            path.write_bytes(encode(b'2 \xff 3'))
            with self.assertRaises(UnicodeDecodeError):
                self.ids(str(path), 8)
        self.assertEqual(self.calls, [])

    def test_cleanup_uses_builtin_notes_for_opaque_primary(self):
        namespace, _ = api(namespace=True)
        self.path.write_bytes(b'fixture')
        raw = self.path.open('rb')

        class Opaque(OSError):
            def add_note(self, *args):
                raise AssertionError('overridden hook must not run')

            def __str__(self):
                raise AssertionError('opaque exception formatting must not run')

        class Raw:
            def fileno(self):
                return raw.fileno()

            def close(self):
                raw.close()
                raise Opaque()

        namespace['open'] = lambda *args, **kwargs: Raw()
        primary = Opaque()
        with self.assertRaises(Opaque) as caught:
            with namespace['_draft_id_stream'](self.path):
                raise primary
        self.assertIs(caught.exception, primary)
        self.assertEqual(primary.__notes__, ['draft vocabulary cleanup failed: Opaque'])

    def test_opaque_first_cleanup_is_preserved_and_all_owners_closed(self):
        namespace, _ = api(namespace=True)
        path = self.path.with_suffix('.gz')
        path.write_bytes(b'fixture')
        raw = path.open('rb')
        closed = []

        class Opaque(OSError):
            def add_note(self, *args):
                raise AssertionError('overridden note hook must not run')

            def __str__(self):
                raise AssertionError('opaque formatting must not run')

        first = Opaque()

        class Raw:
            def fileno(self):
                return raw.fileno()

            def close(self):
                raw.close()
                closed.append('raw')
                raise Opaque()

        class Decoder:
            def close(self):
                closed.append('decoder')
                raise first

        namespace['open'] = lambda *args, **kwargs: Raw()
        namespace['gzip'] = SimpleNamespace(GzipFile=lambda **kw: Decoder())
        with self.assertRaises(Opaque) as caught:
            with namespace['_draft_id_stream'](path):
                pass
        self.assertIs(caught.exception, first)
        self.assertEqual(closed, ['decoder', 'raw'])
        self.assertEqual(first.__notes__, ['additional draft vocabulary cleanup failed: Opaque'])

    def test_public_default_matches_independent_exact_decimal_set(self):
        source = SOURCE.with_name('draft_vocab.txt')
        expected = sorted({int(token) for line in source.read_text().splitlines()
                           for token in line.split('#', 1)[0].split() if int(token) < 248320})
        self.assertEqual(self.ids('default', 248320), expected)
        self.assertEqual(len(expected), 79591)


if __name__ == '__main__':
    unittest.main()

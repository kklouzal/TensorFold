"""Real ordinary-file atomic report replacement/failure controls; no SDK."""
import importlib.util
import json
import os
from pathlib import Path
import stat
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('owned_benchmark_output', ROOT / 'tools/benchmark_output.py')
output = importlib.util.module_from_spec(spec)
spec.loader.exec_module(output)


class Output(unittest.TestCase):
    def test_complete_report_replaces_old_and_preserves_existing_mode(self):
        with tempfile.TemporaryDirectory(prefix='owned-benchmark-output-') as directory:
            path = Path(directory) / 'report.json'
            path.write_text('previous')
            path.chmod(0o640)
            output.write_json(path, {'report': ['caf\u00e9', 17]})
            self.assertEqual(json.loads(path.read_text()), {'report': ['caf\u00e9', 17]})
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o640)
            self.assertEqual(list(path.parent.iterdir()), [path])

    def test_new_report_is_private_and_rejects_symlink_or_directory(self):
        with tempfile.TemporaryDirectory(prefix='owned-benchmark-output-') as directory:
            root = Path(directory)
            path = root / 'report.json'
            output.write_json(path, {'report': 17})
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
            link = root / 'link.json'
            link.symlink_to(path)
            for bad in (link, root):
                with self.assertRaises(ValueError):
                    output.write_json(bad, {'bad': 99})
            self.assertEqual(json.loads(path.read_text()), {'report': 17})

    def test_serialization_or_before_replacement_failure_keeps_previous_report(self):
        for kind in ('serialize', 'file_sync', 'replace', 'interrupt'):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory(prefix='owned-benchmark-output-') as directory:
                path = Path(directory) / 'report.json'
                path.write_text('previous')
                primary = KeyboardInterrupt() if kind == 'interrupt' else OSError('owned publication fault')
                name = 'fsync' if kind == 'file_sync' else 'replace'
                if kind == 'serialize':
                    with self.assertRaises(ValueError):
                        output.write_json(path, {'invalid': float('nan')})
                else:
                    with patch.object(output.os, name, side_effect=primary), self.assertRaises(BaseException) as caught:
                        output.write_json(path, {'report': 17})
                    self.assertIs(caught.exception, primary)
                self.assertEqual(path.read_text(), 'previous')
                self.assertEqual(list(path.parent.iterdir()), [path])

    def test_after_replacement_directory_failure_reports_committed_valid_document(self):
        with tempfile.TemporaryDirectory(prefix='owned-benchmark-output-') as directory:
            path = Path(directory) / 'report.json'
            path.write_text('previous')
            primary, calls = OSError('directory fsync failed'), []
            original = output.os.fsync
            def fsync(descriptor):
                calls.append(descriptor)
                if len(calls) == 2:
                    raise primary
                original(descriptor)
            with patch.object(output.os, 'fsync', fsync), self.assertRaises(OSError) as caught:
                output.write_json(path, {'report': 17})
            self.assertIs(caught.exception, primary)
            self.assertTrue(any('was replaced' in note for note in primary.__notes__))
            self.assertEqual(json.loads(path.read_text()), {'report': 17})
            self.assertEqual(list(path.parent.iterdir()), [path])

    def test_directory_sync_primary_survives_actual_closed_fd_secondary(self):
        with tempfile.TemporaryDirectory(prefix='owned-benchmark-output-') as directory:
            path = Path(directory) / 'report.json'
            path.write_text('previous')
            primary, secondary = OSError('directory sync failed'), OSError('directory close failed')
            real_sync, real_close = os.fsync, os.close
            def sync(descriptor):
                if stat.S_ISDIR(os.fstat(descriptor).st_mode):
                    raise primary
                real_sync(descriptor)
            def close(descriptor):
                is_directory = stat.S_ISDIR(os.fstat(descriptor).st_mode)
                real_close(descriptor)
                if is_directory:
                    raise secondary
            with patch.object(output.os, 'fsync', sync), patch.object(output.os, 'close', close), \
                    self.assertRaises(OSError) as caught:
                output.write_json(path, {'report': 17})
            self.assertIs(caught.exception, primary)
            self.assertIs(primary.__cause__, secondary)
            self.assertTrue(any('was replaced' in note for note in primary.__notes__))
            self.assertEqual(json.loads(path.read_text()), {'report': 17})

    def test_serialization_primary_survives_actual_stream_close_secondary(self):
        with tempfile.TemporaryDirectory(prefix='owned-benchmark-output-') as directory:
            path = Path(directory) / 'report.json'
            path.write_text('previous')
            primary, secondary = TypeError('serialization failed'), OSError('stream close failed')
            real_open = output.os.fdopen
            def fdopen(*args, **kwargs):
                stream = real_open(*args, **kwargs)
                class Owner:
                    def close(self):
                        stream.close()
                        raise secondary
                return Owner()
            with patch.object(output.os, 'fdopen', fdopen), patch.object(output.json, 'dump', side_effect=primary), \
                    self.assertRaises(TypeError) as caught:
                output.write_json(path, {'report': 17})
            self.assertIs(caught.exception, primary)
            self.assertIs(primary.__cause__, secondary)
            self.assertEqual(path.read_text(), 'previous')
            self.assertEqual(list(path.parent.iterdir()), [path])

    def test_actual_file_descriptor_close_failure_is_not_retried(self):
        with tempfile.TemporaryDirectory(prefix='owned-benchmark-output-') as directory:
            path = Path(directory) / 'report.json'
            path.write_text('previous')
            primary, calls = OSError('raw file close failed after close'), []
            real_close = output.os.close
            def close(descriptor):
                calls.append(descriptor)
                real_close(descriptor)
                raise primary
            with patch.object(output.os, 'close', close), self.assertRaises(OSError) as caught:
                output.write_json(path, {'report': 17})
            self.assertIs(caught.exception, primary)
            self.assertEqual(len(calls), 1)
            self.assertEqual(path.read_text(), 'previous')
            self.assertEqual(list(path.parent.iterdir()), [path])

    def test_cleanup_failure_preserves_primary_and_retains_exact_owned_temporary(self):
        with tempfile.TemporaryDirectory(prefix='owned-benchmark-output-') as directory:
            path = Path(directory) / 'report.json'
            path.write_text('previous')
            primary, cleanup = OSError('replace failed'), PermissionError('owned unlink failed')
            original = Path.unlink
            def unlink(temporary):
                if temporary.name.endswith('.tmp'):
                    raise cleanup
                return original(temporary)
            with patch.object(output.os, 'replace', side_effect=primary), patch.object(Path, 'unlink', unlink), \
                    self.assertRaises(OSError) as caught:
                output.write_json(path, {'report': 17})
            self.assertIs(caught.exception, primary)
            self.assertIs(caught.exception.__cause__, cleanup)
            self.assertTrue(primary.__notes__)
            owned = [p for p in path.parent.iterdir() if p != path]
            self.assertEqual(len(owned), 1)
            self.assertTrue(owned[0].name.startswith('.report.json.'))
            self.assertEqual(path.read_text(), 'previous')
            owned[0].unlink()


if __name__ == '__main__':
    unittest.main()

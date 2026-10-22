"""Real ordinary-file atomic report replacement/failure controls; no SDK."""
import importlib.util
import json
import os
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('owned_benchmark_output', ROOT / 'tools/benchmark_output.py')
output = importlib.util.module_from_spec(spec)
sys.path.insert(0, str(ROOT / 'tools'))
try:
    spec.loader.exec_module(output)
finally:
    sys.path.pop(0)


def retained_roots(error):
    """Include the documented operation-owned traceback roots on exhaustion."""
    pending, seen = [error], []
    while pending:
        item = pending.pop()
        if item is None or any(item is previous for previous in seen):
            continue
        seen.append(item)
        pending.extend((item.__cause__, item.__context__))
        if isinstance(item, BaseExceptionGroup):
            pending.extend(item.exceptions)
        trace = item.__traceback__
        while trace is not None:
            if trace.tb_frame.f_code in (output.write_json.__code__, output.raise_failures.__code__):
                state = trace.tb_frame.f_locals
                pending.extend(state.get(key) for key in ('prior_cause', 'prior_context', 'native_cause', 'native_context'))
                pending.extend(state.get('errors', ()))
            trace = trace.tb_next
    return seen


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
                    @property
                    def closed(self):
                        return stream.closed

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
            original = os.unlink
            def unlink(temporary):
                if Path(temporary).name.endswith('.tmp'):
                    raise cleanup
                return original(temporary)
            with patch.object(output.os, 'replace', side_effect=primary), patch.object(output.os, 'unlink', unlink), \
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

    def test_native_causes_and_all_cleanup_roots_survive_publication_failure(self):
        with tempfile.TemporaryDirectory(prefix='owned-benchmark-output-') as directory:
            path = Path(directory) / 'report.json'
            path.write_text('previous')
            primary = TypeError('serialization failed')
            cause, context = LookupError('native cause'), RuntimeError('native context')
            primary.__cause__, primary.__context__ = cause, context
            stream_error, descriptor_error, unlink_error = OSError('stream close'), OSError('raw close'), OSError('unlink')
            opened, events = [], []
            original_open, original_close, original_unlink = output.os.fdopen, output.os.close, output.os.unlink

            def fdopen(*args, **kwargs):
                stream = original_open(*args, **kwargs)
                opened.append(stream)

                class Owner:
                    @property
                    def closed(self):
                        return stream.closed

                    def close(self):
                        events.append('stream')
                        stream.close()
                        raise stream_error
                return Owner()

            def close(descriptor):
                events.append('descriptor')
                original_close(descriptor)
                raise descriptor_error

            def unlink(temporary):
                events.append('unlink')
                original_unlink(temporary)
                raise unlink_error

            with patch.object(output.os, 'fdopen', fdopen), patch.object(output.json, 'dump', side_effect=primary), \
                    patch.object(output.os, 'close', close), patch.object(output.os, 'unlink', unlink), \
                    self.assertRaises(TypeError) as caught:
                output.write_json(path, {'report': 17})
            self.assertIs(caught.exception, primary)
            self.assertEqual(events, ['stream', 'descriptor', 'unlink'])
            self.assertTrue(all(stream.closed for stream in opened))
            roots = retained_roots(primary)
            self.assertTrue(all(any(root is item for root in roots)
                                for item in (cause, context, stream_error, descriptor_error, unlink_error)))
            self.assertEqual(path.read_text(), 'previous')
            self.assertEqual(list(path.parent.iterdir()), [path])

    def test_group_exhaustion_happens_after_resource_retirement_and_keeps_raw_roots(self):
        with tempfile.TemporaryDirectory(prefix='owned-benchmark-output-') as directory:
            path = Path(directory) / 'report.json'
            path.write_text('previous')
            primary, native, cleanup = OSError('replace failed'), LookupError('native cause'), OSError('unlink failed')
            allocation = MemoryError('failure group exhausted')
            primary.__cause__ = native
            original = output.os.unlink
            events = []

            def unlink(temporary):
                original(temporary)
                events.append('unlink')
                raise cleanup

            def group(*args):
                self.assertEqual(events, ['unlink'])
                raise allocation

            provider = sys.modules[output.raise_failures.__module__]
            with patch.object(output.os, 'replace', side_effect=primary), patch.object(output.os, 'unlink', unlink), \
                    patch.object(provider, 'BaseExceptionGroup', group, create=True):
                # assertRaises clears completed traceback frames; retain them
                # here because exhaustion explicitly uses frame-owned roots.
                try:
                    output.write_json(path, {'report': 17})
                except OSError as error:
                    caught = error
                else:
                    self.fail('publication failure was not propagated')
            self.assertIs(caught, primary)
            roots = retained_roots(primary)
            self.assertTrue(all(any(root is item for root in roots) for item in (native, cleanup, allocation)))
            self.assertEqual(path.read_text(), 'previous')
            self.assertEqual(list(path.parent.iterdir()), [path])

    def test_failed_commit_annotation_cannot_mask_sync_or_skip_descriptor_close(self):
        with tempfile.TemporaryDirectory(prefix='owned-benchmark-output-') as directory:
            path = Path(directory) / 'report.json'
            path.write_text('previous')
            primary, native = OSError('directory fsync failed'), LookupError('original cause')
            primary.__cause__ = native
            primary.__notes__ = None  # BaseException.add_note must explicitly reject a non-list notes field.
            original_sync, original_close = output.os.fsync, output.os.close
            retired = []

            def sync(descriptor):
                if stat.S_ISDIR(os.fstat(descriptor).st_mode):
                    raise primary
                original_sync(descriptor)

            def close(descriptor):
                is_directory = stat.S_ISDIR(os.fstat(descriptor).st_mode)
                original_close(descriptor)
                retired.append(is_directory)

            with patch.object(output.os, 'fsync', sync), patch.object(output.os, 'close', close), \
                    self.assertRaises(OSError) as caught:
                output.write_json(path, {'report': 17})
            self.assertIs(caught.exception, primary)
            self.assertEqual(retired, [False, True])
            roots = retained_roots(primary)
            self.assertTrue(any(root is native for root in roots))
            self.assertTrue(any(isinstance(root, TypeError) for root in roots))
            self.assertEqual(json.loads(path.read_text()), {'report': 17})
            self.assertEqual(list(path.parent.iterdir()), [path])

    def test_partial_stream_construction_retires_raw_descriptor_and_temporary(self):
        with tempfile.TemporaryDirectory(prefix='owned-benchmark-output-') as directory:
            path = Path(directory) / 'report.json'
            path.write_text('previous')
            primary, events = MemoryError('stream construction exhausted'), []
            original_close, original_unlink = output.os.close, output.os.unlink

            def close(descriptor):
                original_close(descriptor)
                events.append('descriptor')

            def unlink(temporary):
                original_unlink(temporary)
                events.append('unlink')

            with patch.object(output.os, 'fdopen', side_effect=primary), patch.object(output.os, 'close', close), \
                    patch.object(output.os, 'unlink', unlink), self.assertRaises(MemoryError) as caught:
                output.write_json(path, {'report': 17})
            self.assertIs(caught.exception, primary)
            self.assertEqual(events, ['descriptor', 'unlink'])
            self.assertEqual(path.read_text(), 'previous')
            self.assertEqual(list(path.parent.iterdir()), [path])

    def test_unproved_stream_close_retains_its_fd_and_never_retries_or_reuses_it(self):
        for stage in ('normal-close', 'cleanup-close', 'closed-query'):
            with self.subTest(stage=stage), tempfile.TemporaryDirectory(prefix='owned-benchmark-output-') as directory:
                path = Path(directory) / 'report.json'
                path.write_text('previous')
                serialization, closing = TypeError('serialization failed'), OSError('close failed before retirement')
                query = MemoryError('closed lookup exhausted')
                original_open, original_close, original_unlink = output.os.fdopen, output.os.close, output.os.unlink
                owner, events = {}, []

                def fdopen(descriptor, *args, original_open=original_open, owner=owner,
                           stage=stage, query=query, events=events, closing=closing, **kwargs):
                    stream = original_open(descriptor, *args, **kwargs)
                    owner.update(stream=stream, descriptor=descriptor)

                    class Owner:
                        def __getattr__(self, name):
                            return getattr(stream, name)

                        @property
                        def closed(self):
                            if stage == 'closed-query':
                                raise query
                            return stream.closed

                        def close(self):
                            events.append('stream-close')
                            raise closing
                    return Owner()

                def close(descriptor, events=events, original_close=original_close):
                    events.append('raw-close')
                    original_close(descriptor)

                def unlink(temporary, events=events, original_unlink=original_unlink):
                    events.append('unlink')
                    original_unlink(temporary)

                try:
                    with patch.object(output.os, 'fdopen', fdopen), patch.object(output.os, 'close', close), \
                            patch.object(output.os, 'unlink', unlink):
                        try:
                            if stage == 'normal-close':
                                output.write_json(path, {'report': 17})
                            else:
                                with patch.object(output.json, 'dump', side_effect=serialization):
                                    output.write_json(path, {'report': 17})
                        except (TypeError, OSError) as error:
                            caught = error
                        else:
                            self.fail('unproved retirement was silently accepted')
                    self.assertIs(caught, closing if stage == 'normal-close' else serialization)
                    self.assertEqual(events, ['stream-close', 'unlink'])
                    self.assertFalse(owner['stream'].closed)
                    os.fstat(owner['descriptor'])  # The raw FD is still reserved for the unretired stream.
                    roots = retained_roots(caught)
                    self.assertTrue(any(root is closing for root in roots))
                    if stage == 'closed-query':
                        self.assertTrue(any(root is query for root in roots))
                    self.assertTrue(any('raw descriptor remains owned' in note for note in caught.__notes__))
                    self.assertEqual(path.read_text(), 'previous')
                    self.assertEqual(list(path.parent.iterdir()), [path])
                finally:
                    # The test is the explicit recovery owner; production
                    # callers must likewise retain the error or retire process.
                    owner['stream'].close()
                    original_close(owner['descriptor'])

    def test_direct_and_package_clients_remain_stdlib_only_without_tensorfold(self):
        script = r'''
import builtins, importlib, json, pathlib, sys, tempfile
root, mode = pathlib.Path(sys.argv[1]), sys.argv[2]
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name == 'tensorfold' or name.startswith('tensorfold.'):
        raise AssertionError('standalone publisher imported TensorFold')
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
sys.path.insert(0, str(root if mode == 'package' else root / 'tools'))
writer = importlib.import_module('tools.benchmark_output' if mode == 'package' else 'benchmark_output')
with tempfile.TemporaryDirectory() as directory:
    target = pathlib.Path(directory) / 'report.json'
    writer.write_json(target, {'complete': True})
    assert json.loads(target.read_text()) == {'complete': True}
    try:
        writer.write_json(target, {'bad': float('nan')})
    except ValueError:
        pass
    else:
        raise AssertionError('invalid serialization silently passed')
    assert json.loads(target.read_text()) == {'complete': True}
assert pathlib.Path(writer.__file__).resolve() == root / 'tools/benchmark_output.py'
provider = sys.modules[writer.raise_failures.__module__]
assert pathlib.Path(provider.__file__).resolve() == root / 'tools/worker_lifetime.py'
assert not any(name == 'tensorfold' or name.startswith('tensorfold.') for name in sys.modules)
print('stdlib-only-' + mode)
'''
        for mode in ('direct', 'package'):
            with self.subTest(mode=mode):
                result = subprocess.run([sys.executable, '-I', '-c', script, str(ROOT), mode],
                                        capture_output=True, text=True, timeout=10, check=False)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout.strip(), 'stdlib-only-' + mode)


if __name__ == '__main__':
    unittest.main()

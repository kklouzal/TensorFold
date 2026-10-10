"""Actual EXL3 ownership code with labeled stdlib mapping/tensor carriers.

Real files, mmap spans, executor callbacks and close ordering are exercised.
There is no NumPy/Torch/native numerical or installed-provider credit here;
test_exl3_ngram_table performs those complementary real-runtime checks.
"""
from __future__ import annotations

import ast
import mmap
from pathlib import Path
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import test_host_table_lifetime_control as mapped

SOURCE = mapped.ROOT / 'src/tensorfold/families/qwen4_exp/cuda/exl3_pack.py'
LOADER = SOURCE.with_name('exl3.py')


class Mapping:
    """Opaque dtype widths and a real stdlib mmap; no ndarray arithmetic."""
    def __init__(self, path, *, dtype, mode, offset=0, shape=None):
        if mode != 'r':
            raise ValueError('read-only source fixture')
        with open(path, 'rb') as file:
            self.span = mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ)
        self.filename, self.offset = str(path), offset
        self.nbytes = len(self.span) if shape is None else shape[0] * shape[1] * dtype

    def bytes(self):
        return self.span[self.offset:self.offset + self.nbytes]


class Opaque:
    def to(self, _):
        return self

    def contiguous(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return (0, 1)


def namespace():
    scope = mapped.definitions('_MappedTable', 'HostTable', '_release_mapped', '_unlock_pins', '_note',
                               '_retain_prefetch_lifetime', '_release_unpublished_file', '_retire_local_file',
                               '_release_prefetch_files', '_joined_copy', '_prefetch_file', '_prefetch')
    scope.update(__name__='tensorfold.families.qwen4_exp.cuda.exl3_pack_source',
                 __package__='tensorfold.families.qwen4_exp.cuda',
                 np=SimpleNamespace(ndarray=object, int16=2, uint8=1, int64=8, memmap=Mapping,
                                    array=lambda value, dtype: list(value)), torch=SimpleNamespace(float16='opaque-f16'))
    tree = ast.parse(SOURCE.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'NgramTable')
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(SOURCE), 'exec',
                 flags=__import__('__future__').annotations.compiler_flag), scope)
    return scope


class Ownership(unittest.TestCase):
    def namespace(self):
        scope = namespace()
        host = ModuleType('tensorfold.families.qwen4_exp.host_table')
        host.HostTable = scope['HostTable']
        providers = patch.dict(sys.modules, {host.__name__: host})
        providers.start()
        self.addCleanup(providers.stop)
        return scope

    def table(self, path, scope):
        rows, words = 8, 41
        path.write_bytes(bytes((i * 17) % 256 for i in range(2 * rows * words)))
        pack = SimpleNamespace(dir=path.parent, entry={'t.trellis': (path.name, 0, path.stat().st_size,
                                                                 'I16', [rows, words])}.__getitem__,
                               get=lambda _: Opaque())
        return scope['NgramTable'](pack, 't.', 2, 'opaque-device')

    def test_actual_constructor_prefetch_owner_and_owned_mapping_retirement(self):
        scope = self.namespace()
        with tempfile.TemporaryDirectory() as directory:
            table = self.table(Path(directory) / 'rows', scope)
            borrowed = table.words[0]
            expected = borrowed.bytes()
            self.assertGreaterEqual(table.prefetch(2), 0)
            table.close()
            self.assertTrue(table._life.closed)
            self.assertEqual(table.words, [])
            self.assertEqual(table.maps, [])
            self.assertEqual(borrowed.bytes(), expected)
            with self.assertRaisesRegex(ValueError, 'closed or unusable'):
                table.gather([0])
            with self.assertRaisesRegex(ValueError, 'closed or unusable'):
                table.lock()
            borrowed.span.close()

    def test_real_prefetch_read_borrower_keeps_maps_until_close_drains(self):
        scope = self.namespace()
        with tempfile.TemporaryDirectory() as directory:
            table = self.table(Path(directory) / 'rows', scope)
            entered, release = threading.Event(), threading.Event()
            original = scope['_prefetch']

            def read(arrays, workers):
                entered.set()
                if not release.wait(10):
                    raise TimeoutError('source fixture release')
                return original(arrays, workers)

            scope['_prefetch'] = read
            with ThreadPoolExecutor(2) as pool:
                pending = pool.submit(table.prefetch, 2)
                self.assertTrue(entered.wait(3))
                close = pool.submit(table.close)
                try:
                    with table._life.condition:
                        for _ in range(100):
                            if table._life.closing:
                                break
                            table._life.condition.wait(.01)
                        self.assertTrue(table._life.closing)
                    self.assertFalse(close.done())
                    self.assertTrue(table.words and table.maps)
                finally:
                    release.set()
                self.assertGreaterEqual(pending.result(3), 0)
                close.result(3)
            self.assertTrue(table._life.closed)
            self.assertFalse(table.words or table.maps)

    def test_constructor_metadata_failure_closes_every_acquired_mapping_owner(self):
        scope = self.namespace()
        captured = []
        actual = scope['NgramTable']._arrays

        def record(self):
            arrays = actual(self)
            captured.append((self, arrays))
            return arrays

        primary = OSError('real source metadata read refusal')
        with patch.object(scope['NgramTable'], '_arrays', record), tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'rows'
            path.write_bytes(bytes(8 * 41 * 2))

            def fail(_):
                raise primary

            pack = SimpleNamespace(dir=path.parent, entry={'t.trellis': (path.name, 0, path.stat().st_size,
                                                                     'I16', [8, 41])}.__getitem__, get=fail)
            with self.assertRaises(OSError) as failure:
                scope['NgramTable'](pack, 't.', 2, 'opaque-device')
            self.assertIs(failure.exception, primary)
            self.assertTrue(captured)
            self.assertTrue(all(table._life.closed and not arrays for table, arrays in captured))

    def test_first_mapping_and_metadata_device_copy_failures_roll_back_owned_lists(self):
        for mode in ('second_map', 'metadata_copy'):
            with self.subTest(mode=mode):
                scope, captured, calls = self.namespace(), [], []
                arrays = scope['NgramTable']._arrays
                primary = MemoryError('source partial acquisition')

                def record(self):
                    value = arrays(self)
                    captured.append((self, value))
                    return value

                def memmap(*args, **kwargs):
                    calls.append(True)
                    if mode == 'second_map' and len(calls) == 2:
                        raise primary
                    return Mapping(*args, **kwargs)

                class DeviceCopy(Opaque):
                    def to(self, target):
                        if target == 'opaque-device':
                            raise primary
                        return self

                scope['np'].memmap = memmap
                with patch.object(scope['NgramTable'], '_arrays', record), tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / 'rows'
                    path.write_bytes(bytes(8 * 41 * 2))
                    pack = SimpleNamespace(dir=path.parent, entry={'t.trellis': (path.name, 0, path.stat().st_size,
                                                                             'I16', [8, 41])}.__getitem__,
                                           get=lambda _: DeviceCopy())
                    with self.assertRaises(MemoryError) as failure:
                        scope['NgramTable'](pack, 't.', 2, 'opaque-device')
                    self.assertIs(failure.exception, primary)
                    self.assertTrue(all(table._life.closed and not values for table, values in captured))

    def test_public_loader_publishes_and_failed_loader_retires_actual_PLE_journal(self):
        tree = ast.parse(LOADER.read_text())
        load = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'load')
        cleanup = mapped.stdlib_module('exl3_source_cleanup', 'src/tensorfold/cleanup.py')
        cleanup_module = ModuleType('tensorfold.cleanup')
        cleanup_module.finish, cleanup_module.rollback = cleanup.finish, cleanup.rollback
        cleanup_module.raise_failures = cleanup.raise_failures
        with patch.dict(sys.modules, {'tensorfold.cleanup': cleanup_module}):
            ple = mapped.stdlib_module('exl3_source_ple_owner', 'src/tensorfold/families/qwen4_exp/ple_lifetime.py')
        weight_types = ModuleType('tensorfold.families.qwen4_exp.cuda.weight_types')
        weight_types.Config = SimpleNamespace(read=lambda *a, **k: SimpleNamespace(vocab=32))
        weight_types.draft_token_ids = lambda *a: None
        owners, closed = [], []

        def fake_load(*args, _ple_tables, **kwargs):
            owners.append(_ple_tables)
            _ple_tables.acquire(SimpleNamespace(close=lambda: closed.append(True)))
            return SimpleNamespace(meta={})

        scope = {'__name__': 'tensorfold.families.qwen4_exp.cuda.exl3_source',
                 '__package__': 'tensorfold.families.qwen4_exp.cuda', 'PLETables': ple.PLETables, '_load': fake_load,
                 'torch': SimpleNamespace(device=lambda _: SimpleNamespace(type='cpu'))}
        exec(compile(ast.Module(body=[load], type_ignores=[]), str(LOADER), 'exec',
                     flags=__import__('__future__').annotations.compiler_flag), scope)
        modules = {'tensorfold.families.qwen4_exp.cuda.weight_types': weight_types, 'tensorfold.cleanup': cleanup_module}
        with patch.dict(sys.modules, modules):
            reads = []
            result = scope['load']('/opaque-model', device='cpu', table_reads=reads)
            self.assertIs(result.meta['ple_tables'], owners[-1])
            self.assertIs(owners[-1].reads, reads)
            result.meta['ple_tables'].close(lambda: None)
            self.assertEqual(closed, [True])
            primary = KeyboardInterrupt('source load failure after acquire')

            def fail(*args, _ple_tables, **kwargs):
                owners.append(_ple_tables)
                _ple_tables.acquire(SimpleNamespace(close=lambda: closed.append(True)))
                raise primary

            scope['_load'] = fail
            with self.assertRaises(KeyboardInterrupt) as failure:
                scope['load']('/opaque-model', device='cpu')
            self.assertIs(failure.exception, primary)
            self.assertTrue(owners[-1].closed)
            self.assertEqual(closed, [True, True])

            cleanup_error = OSError('source table close failure')

            def fail_close():
                raise cleanup_error

            def fail_retirement(*args, _ple_tables, **kwargs):
                owners.append(_ple_tables)
                _ple_tables.acquire(SimpleNamespace(close=fail_close))
                raise primary

            scope['_load'] = fail_retirement
            with self.assertRaises(KeyboardInterrupt) as failure:
                scope['load']('/opaque-model', device='cpu')
            self.assertIs(failure.exception, primary)
            self.assertFalse(owners[-1].closed)
            retained = BaseException.__dict__['__dict__'].__get__(primary)['_tensorfold_retained_owners']
            self.assertTrue(any(owner[0] is owners[-1] for owner in retained))
            self.assertIs(BaseException.__cause__.__get__(primary), cleanup_error)


if __name__ == '__main__':
    unittest.main()
